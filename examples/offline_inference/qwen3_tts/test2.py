import os
from typing import NamedTuple  # noqa: UP035

import soundfile as sf

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from vllm import SamplingParams

from vllm_omni import Omni


class QueryResult(NamedTuple):
    """Container for a prepared Omni request."""

    inputs: dict
    model_name: str


# new
def get_base_query(
    ref_audios: list[str],
    ref_texts: list[str],
    target_texts: list[str],
    target_langs: list[str],
):
    inputs = []
    for target_text, target_lang, ref_audio, ref_text in zip(
        target_texts,
        target_langs,
        ref_audios,
        ref_texts,
    ):
        prompt = f"<|im_start|>assistant\n{target_text}<|im_end|>\n<|im_start|>assistant\n"
        print(prompt)
        inputs.append(
            {
                "prompt": prompt,
                "additional_information": {
                    "task_type": ["Base"],
                    "ref_audio": [ref_audio],
                    "ref_text": [ref_text],
                    "text": [target_text],
                    "language": [target_lang],
                    "x_vector_only_mode": [False],
                    "max_new_tokens": [8192],
                },
            }
        )

    return QueryResult(
        inputs=inputs,
        model_name="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    )


def main():
    omni = Omni(
        model="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        stage_configs_path="/lustre/users/rkoshkin/vllm-omni/vllm_omni/model_executor/stage_configs/qwen3_tts.yaml",
        log_stats=True,
        stage_ibnit_timeout=300,
    )

    target_texts = [
        "Welcome to another episode of Out of the Pods.",
        "I'm Deep T. And I'm Natalie.",
        "And happy Wednesday.",
        "You know, we said last week that this episode is going to be about our recap of Perfect Match Season 2, Episodes 1 through 6, which we will get into.",
        "Lots of thoughts.",
        "Actually, almost no thoughts because...",
        "This is not a great season.",
        "It's just not off to a good start.",
        "I feel like I lost some brain cells watching it.",
        "Oh, 100%.",
    ]
    # ref_audios = ["https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone_2.wav"] * len(target_texts)
    # ref_texts = [
    #     "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you.",
    # ] * len(target_texts)
    ref_audios = ["/lustre/users/rkoshkin/s2st/bak/trump.mp3"] * len(target_texts)
    ref_texts = [
        "because of it. Look, we were ripped off by almost every country in the world. If you look at the surpluses, almost every country in the world that did business with us, our people were stupid. And I blame presidents for it because they're ultimately the leader. Uh we were being ripped off by almost every single country in the world had massive some massive surpluses. China had hundreds of billions of dollars in surpluses with the United States. They rebuilt China. They rebuilt the army. We built China's army by allowing that to happen. I have a great relationship with President Xi, but he respects our country now. Now, what we've done, I charged China a 20% tariff as a penalty for sending fentinol in. And that was 20 times more than they could make by selling fentanol."
    ] * len(target_texts)
    target_langs = ["English"] * len(target_texts)

    query_result = get_base_query(ref_audios, ref_texts, target_texts, target_langs)

    sampling_params = SamplingParams(
        temperature=0.9,
        top_p=1.0,
        top_k=50,
        max_tokens=8192,
        seed=42,
        detokenize=False,
        repetition_penalty=1.05,
    )

    sampling_params_list = [
        sampling_params,
    ]

    output_dir = "/lustre/users/rkoshkin/vllm-omni/examples/offline_inference/qwen3_tts/output"
    os.makedirs(output_dir, exist_ok=True)

    omni_generator = omni.generate(query_result.inputs, sampling_params_list)
    for stage_outputs in omni_generator:
        for output in stage_outputs.request_output:
            request_id = output.request_id
            audio_tensor = output.outputs[0].multimodal_output["audio"].clone()
            print(f"audio_tensor: {audio_tensor.shape}")
            output_wav = os.path.join(output_dir, f"output_{request_id}.wav")
            audio_samplerate = output.outputs[0].multimodal_output["sr"].item()
            # Convert to numpy array and ensure correct format
            audio_numpy = audio_tensor.float().detach().cpu().numpy()

            # Ensure audio is 1D (flatten if needed)
            if audio_numpy.ndim > 1:
                audio_numpy = audio_numpy.flatten()

            # Save audio file with explicit WAV format
            sf.write(output_wav, audio_numpy, samplerate=audio_samplerate, format="WAV")
            print(f"Request ID: {request_id}, Saved audio to {output_wav}")


if __name__ == "__main__":
    main()
