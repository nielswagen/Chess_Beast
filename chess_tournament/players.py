from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict
import chess
import random
import requests
import torch
import re
import time
import os

from huggingface_hub import InferenceClient
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig

class Player(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def get_move(self, fen: str) -> Optional[str]:
        pass


class RandomPlayer(Player):
    def get_move(self, fen: str) -> Optional[str]:
        board = chess.Board(fen)
        moves = list(board.legal_moves)
        return random.choice(moves).uci() if moves else None


class EnginePlayer(Player):
    """
    EnginePlayer now behaves like ANY Player:
    Input: FEN
    Output: move string (UCI) | "__NO_MOVES__" | None

    Internal failures are NOT visible to Game.
    """

    def __init__(
        self,
        name: str,
        blunder_rate: float = 0.0,
        ponder_rate: float = 0.0,
        base_delay: float = 0.9,
        enable_cache: bool = True,
    ):
        super().__init__(name)

        assert 0.0 <= blunder_rate <= 1.0
        assert 0.0 <= ponder_rate <= 1.0
        assert blunder_rate + ponder_rate <= 1.0

        self.blunder_rate = blunder_rate
        self.ponder_rate = ponder_rate
        self.base_delay = base_delay
        self.enable_cache = enable_cache

        self.api_key = os.environ.get("RAPIDAPI_KEY")
        if not self.api_key:
            raise ValueError("RAPIDAPI_KEY must be set")

        self.url = "https://chess-stockfish-16-api.p.rapidapi.com/chess/api"
        self.headers = {
            "x-rapidapi-key": self.api_key,
            "x-rapidapi-host": "chess-stockfish-16-api.p.rapidapi.com",
            "Content-Type": "application/x-www-form-urlencoded"
        }

        self.cache: Dict[str, Tuple[str, Optional[str]]] = {}

    def _sleep(self):
        time.sleep(self.base_delay)

    def _random_legal_from_fen(self, fen: str) -> Optional[str]:
        try:
            board = chess.Board(fen)
        except Exception:
            return None
        legal = list(board.legal_moves)
        if not legal:
            return None
        return random.choice(legal).uci()

    def _choose_move(self, best: str, ponder: Optional[str], fen: str) -> str:
        r = random.random()

        if r < self.blunder_rate:
            rm = self._random_legal_from_fen(fen)
            return rm if rm else best

        if r < self.blunder_rate + self.ponder_rate:
            return ponder if ponder else best

        return best

    def get_move(self, fen: str) -> Optional[str]:

        # CACHE
        if self.enable_cache and fen in self.cache:
            best, ponder = self.cache[fen]
            return self._choose_move(best, ponder, fen)

        self._sleep()

        try:
            r = requests.post(self.url, data={"fen": fen}, headers=self.headers, timeout=10)
            if r.status_code != 200:
                return None

            j = r.json()

        except Exception:
            return None

        # Engine says no moves
        result_field = j.get("result")
        if isinstance(result_field, str) and "bestmove (none)" in result_field.lower():

            rm = self._random_legal_from_fen(fen)
            if rm is None:
                return "__NO_MOVES__"

            return rm  # Game will treat as normal move

        best = j.get("bestmove")
        ponder = j.get("ponder")

        if not best:
            return None

        if self.enable_cache:
            self.cache[fen] = (best, ponder if ponder else None)

        return self._choose_move(best, ponder if ponder else None, fen)

class LMPlayer(Player):
    def __init__(
        self,
        name: str,
        model_id: str = "mistralai/Mistral-7B-Instruct-v0.2",
        quantization: Optional[str] = "4bit",
        temperature: float = 0.1,
        max_new_tokens: int = 6,
        retries: int = 5
    ):
        super().__init__(name)

        self.model_id = model_id
        self.quantization = quantization
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.retries = retries

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[{self.name}] Loading {self.model_id} on {self.device}")
        print(f"[{self.name}] Quantization mode: {self.quantization}")

        # -------------------------
        # Quantization config
        # -------------------------
        quant_config = None

        if quantization == "4bit":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

        elif quantization == "8bit":
            quant_config = BitsAndBytesConfig(
                load_in_8bit=True
            )

        elif quantization is None:
            quant_config = None

        else:
            raise ValueError("quantization must be one of: None, '8bit', '4bit'")

        # -------------------------
        # Tokenizer
        # -------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # -------------------------
        # Config
        # -------------------------
        config = AutoConfig.from_pretrained(model_id)
        config.pad_token_id = self.tokenizer.pad_token_id

        # -------------------------
        # Model loading
        # -------------------------
        if quant_config:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                config=config,
                quantization_config=quant_config,
                device_map="auto"
            )
        else:
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                config=config,
                dtype=dtype,
                device_map="auto"
            )

        # -------------------------
        # UCI regex
        # -------------------------
        self.uci_re = re.compile(r"\b[a-h][1-8][a-h][1-8][qrbn]?\b")

    def _build_prompt(self, fen: str) -> str:
        return f"""You are a chess engine.

Your task is to output the BEST LEGAL MOVE for the given chess position.

STRICT OUTPUT RULES:
- Output EXACTLY ONE move
- UCI format ONLY (examples: e2e4, g1f3, e7e8q)
- NO explanations
- NO punctuation
- NO extra text

Examples:

FEN: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
Move: e2e4

FEN: r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3
Move: f1b5

FEN: rnbqkb1r/pppp1ppp/5n2/4p3/1P6/5NP1/P1PPPP1P/RNBQKB1R b KQkq - 0 3
Move: e5e4

Now evaluate this position:

FEN: {fen}
Move:"""

    def _extract_move(self, text: str) -> Optional[str]:
        match = self.uci_re.search(text)
        return match.group(0) if match else None

    def get_move(self, fen: str) -> Optional[str]:
        prompt = self._build_prompt(fen)

        for attempt in range(1, self.retries + 1):

            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=self.temperature,
                    pad_token_id=self.tokenizer.pad_token_id
                )

            decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            if decoded.startswith(prompt):
                decoded = decoded[len(prompt):]

            move = self._extract_move(decoded)

            if move:
                return move

        return None

class SmolPlayer(Player):
    """
    LLMAPIPlayer using InferenceClient.chat_completion()
    Compatible with chat/instruct models.
    """

    UCI_REGEX = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b", re.IGNORECASE)

    def __init__(
        self,
        name: str,
        model_id: str = 'moonshotai/Kimi-K2-Instruct',
        temperature: float = 0.2,
        max_tokens: int = 32,
    ):
        super().__init__(name)

        self.client = InferenceClient(
            model=model_id,
            token=os.environ.get("HF_TOKEN")
        )

        self.temperature = temperature
        self.max_tokens = max_tokens

    def _build_prompt(self, fen: str) -> str:
        return f"""You are a chess engine.

Your task is to output the BEST LEGAL MOVE for the given chess position.

STRICT OUTPUT RULES:
- Output EXACTLY ONE move
- UCI format ONLY (examples: e2e4, g1f3, e7e8q)
- NO explanations
- NO punctuation
- NO extra text

Examples:

FEN: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
Move: e2e4

FEN: r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3
Move: f1b5

FEN: rnbqkb1r/pppp1ppp/5n2/4p3/1P6/5NP1/P1PPPP1P/RNBQKB1R b KQkq - 0 3
Move: e5e4

Now evaluate this position:

FEN: {fen}
Move:"""

    def _extract_uci(self, text: str):
        if not text:
            return None

        match = self.UCI_REGEX.search(text)
        return match.group(1).lower() if match else None

    def get_move(self, fen: str):

        prompt = self._build_prompt(fen)

        try:
            response = self.client.chat_completion(
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            text = response.choices[0].message.content

            return self._extract_uci(text)

        except Exception as e:
            # Optional debug:
            print(f"[{self.name}] API error:", e)
            return None
          
