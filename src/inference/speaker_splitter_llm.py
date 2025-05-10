from typing import List, Tuple
from pathlib import Path
from .llm_utils import load_model, generate_response


class SpeakerDetectorLLM:
    def __init__(self, model_path: str | Path, n_gpu_layers: int | None = None, n_ctx: int = 4096):
        """
        Initialize the speaker detector with a GGUF model.
        
        Args:
            model_path: Path to the .gguf model file
            n_gpu_layers: Number of layers to place on GPU
            n_ctx: Context window size
        """
        self.llm = load_model(model_path, n_gpu_layers, n_ctx=n_ctx)

    def detect_speaker_and_location(self, chunk: str) -> Tuple[bool, str, str]:
        """
        Analyzes a text chunk to detect speaker changes.
        
        Returns:
        - speaker_change_detected: bool
        - new_speaker: str or None
        - change_sentence: str or None
        """
        prompt = (
            "[INST] <<SYS>>Eres un asistente analizando transcripciones de discursos políticos. "
            "Responde en el formato exacto especificado.<</SYS>>\n\n"
            f"Dado el siguiente fragmento de transcripción:\n\n{chunk}\n\n"
            "Tus tareas:\n"
            "- Detecta si se introduce un nuevo orador.\n"
            "- Si es así, proporciona el nombre completo/título del nuevo orador y la oración exacta donde ocurre el cambio.\n"
            "- Si no, responde \"No speaker change\".\n\n"
            "Formatea tu respuesta exactamente como:\n"
            "[Nombre del Orador]|[Oración de Cambio]\n"
            "o\n"
            "[No speaker change]|[Oración completa][/INST]"
        )
        
        cleaned = generate_response(self.llm, prompt)

        if cleaned.lower() == "no speaker change":
            return False, None, None
        else:
            try:
                speaker, sentence = cleaned.split("|", 1)
                return True, speaker.strip(), sentence.strip()
            except ValueError:
                raise ValueError(f"Unexpected LLM output format: {cleaned}")

class SpeakerAwareSplitter:
    def __init__(self, speaker_detector: SpeakerDetectorLLM):
        """
        speaker_detector: instance of SpeakerDetectorLLM
        """
        self.speaker_detector = speaker_detector

    def split_chunk_on_speaker(self, chunk: str, default_speaker: str) -> List[Tuple[str, str]]:
        """
        Splits a single semantic chunk based on speaker change inside it.
        
        Returns a list of (speaker, text) pairs.
        """
        speaker_change, new_speaker, change_sentence = self.speaker_detector.detect_speaker_and_location(chunk)

        if not speaker_change:
            return [(default_speaker, chunk.strip())]

        idx = chunk.find(change_sentence)

        if idx == -1:
            # Fallback: assign all chunk to the new speaker
            return [(new_speaker, chunk.strip())]
        
        before_text = chunk[:idx].strip()
        after_text = chunk[idx:].strip()

        splits = []
        if before_text:
            splits.append((default_speaker, before_text))
        if after_text:
            splits.append((new_speaker, after_text))
        
        return splits

    def split_chunks(self, semantic_chunks: List[str], starting_speaker: str = "Maestro de Ceremonias") -> List[Tuple[str, str]]:
        """
        Input: List of semantic chunks (strings).
        Output: List of (speaker, text) tuples.
        """
        results = []
        current_speaker = starting_speaker

        for chunk in semantic_chunks:
            splits = self.split_chunk_on_speaker(chunk, current_speaker)
            results.extend(splits)
            if splits:
                current_speaker = splits[-1][0]  # Update last known speaker
        
        return results
