import os
from typing import NamedTuple
import soundfile as sf
from typing import List
from datasets import Dataset, DatasetDict

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from vllm import SamplingParams
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm_omni import Omni

class QueryResult(NamedTuple):
    """Container for a prepared Omni request."""

    inputs: dict
    model_name: str

# new
def get_base_query(
    ref_audios: List[str],
    ref_texts: List[str],
    target_texts: List[str],
    target_langs: List[str],
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

    ds = DatasetDict.load_from_disk("/lustre/users/rkoshkin/s2st/data/s2s/podcast_crawl-enru-dd-full-s2s+sid/")['train']
    target_texts = [ds[0]['src_sent'][i] for i in range(80)]
    ref_audios = ["/lustre/users/rkoshkin/s2st/assets/ru_ref.sample.wav"] * len(target_texts)
    ref_texts = [
        "Привет! С вами Программный Комитет - шоу подкаст-студии Термин-Вокс и IT-конфереции Стачка. Меня зовут Сергей Пихин.В этом подкасте мы обсуждаем главные тренды в IT-индустрии и в смежных областях. Помогают нам в этом топовые эксперты, которые делятся своими знаниями и экспертизой.",
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