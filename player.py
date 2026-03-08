from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple

import chess
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from chess_tournament.players import Player


# instellingen voor de zoekstrategie
@dataclass(frozen=True)
class SearchConfig:
    top_k: int = 5
    opp_top_k: int = 5
    depth: int = 3

    # in het eindspel gaan we iets dieper zoeken
    endgame_depth: int = 4
    endgame_threshold: int = 10

    # veiligheidslimiet voor het aantal zetten dat we evalueren
    max_legal: int = 80


#  waardes voor stukken
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3.2,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}


class TransformerPlayer(Player):

    def __init__(self, name: str,
                 model_id: str = "donquichot/chess_transformer",
                 temperature: float = 0.0):

        super().__init__(name)
        self.model_id = model_id
        self.temperature = temperature
        self.cfg = SearchConfig()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.MAX_LEN = None

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer.padding_side = "left"

        dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
        )

        self.model.to(self.device)
        self.model.eval()

        self.MAX_LEN = self.model.config.n_positions

    def get_move(self, fen: str) -> Optional[str]:
        """
        Hoofd functie die wordt aangeroepen krijgt fen string en returned uci
        """
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None

        # niet te veel zetten evalueren, wordt anders te traag
        if len(legal_moves) > self.cfg.max_legal:
            legal_moves = legal_moves[:self.cfg.max_legal]

        move = self.choose(board)

        if move is not None:
            return move

        # als alles mis gaat kies de beste zet
        return self.best_model_move(board).uci()

    def get_depth(self, board: chess.Board) -> int:
        # bepaal hoe diep we zoeken en in het eindspel zoeken we iets dieper
        piece_count = len(board.piece_map())

        if piece_count <= self.cfg.endgame_threshold:
            return self.cfg.endgame_depth

        return self.cfg.depth

    def choose(self, board: chess.Board) -> Optional[str]:
        # direct kijken of ik in 1 zet kan winnen
        quick = self.winning_capture_or_mate(board)
        if quick is not None:
            return quick

        # anders minimax search
        return self.minmax(board)



    def winning_capture_or_mate(self, board: chess.Board) -> Optional[str]:
         # Check of er een directe mat of iemand makkelijk te pakken is
        for move in board.legal_moves:
            board.push(move)
            is_mate = board.is_checkmate()
            board.pop()

            if is_mate:
                return move.uci()

            if board.is_capture(move):
                victim = board.piece_at(move.to_square)
                attacker = board.piece_at(move.from_square)
                if victim is None or attacker is None:
                    continue
                # alleen slaan als andere pioneetje meer waard is
                victim_val = PIECE_VALUES.get(victim.piece_type, 0)
                attacker_val = PIECE_VALUES.get(attacker.piece_type, 0)

                # alleen interessante captures
                if victim_val > attacker_val:
                    return move.uci()

        return None


    def minmax(self, board: chess.Board) -> Optional[str]:
        # minmax search waarbij we alleen naar de beste zetten van het model kijken
        piece_count = len(board.piece_map())
        endgame = piece_count <= self.cfg.endgame_threshold

        top_k = self.cfg.top_k + (1 if endgame else 0)
        opp_top_k = self.cfg.opp_top_k + (1 if endgame else 0)

        ranked = self.rank_moves(board, top_k)

        if not ranked:
            return None

        # beginnen met beste zetten volgens model
        best_move = ranked[0][0]
        best_value = float("-inf")

        for my_move, my_score in ranked:

            board.push(my_move)

            # als die directe mate geeft kiezen we hem
            if board.is_checkmate():
                board.pop()
                return my_move.uci()

            # herhaling vermijden, zorgt voor gelijkspel
            if board.is_repetition(3):
                board.pop()

                if best_value == float("-inf"):
                    best_value = -1e6
                    best_move = my_move

                continue

            if board.is_stalemate() or board.is_insufficient_material():
                board.pop()

                if 0.0 > best_value:
                    best_value = 0.0
                    best_move = my_move

                continue
            # controleren wat de tegenstander zou doen erna
            opp_ranked = self.rank_moves(board, opp_top_k)
            worst = self.eval_opp_reply(board, opp_ranked, my_score)

            board.pop()

            # kies zet met beste score
            if worst > best_value:
                best_value = worst
                best_move = my_move

        return best_move.uci()


    def eval_opp_reply(self, board: chess.Board, opp_ranked, parent_score: float) -> float:
        if not opp_ranked:
            return self.terminal_value(board)

        worst_score = float("inf")
        for opp_move, opp_neural_score in opp_ranked:
            board.push(opp_move)
            # bekijk alle zetten van de tegenstander
            if self.mate_in_one_opp(board):
                pos_value = float("-1e9")
            else:
                pos_value = (self.evaluate_position(board) + 0.5 * opp_neural_score + 0.2 * parent_score)

            worst_score = min(worst_score, pos_value)
            board.pop()

        return worst_score

    @staticmethod
    def terminal_value(board: chess.Board) -> float:
        if board.is_checkmate():
            return float("-1e9")
        else:
            return 0.0

    @staticmethod
    def mate_in_one_opp(board: chess.Board) -> bool:
        for mv in board.legal_moves:
            board.push(mv)
            is_mate = board.is_checkmate()
            board.pop()
            if is_mate:
                return True
        return False

    # verschillende schaakzetten scores geven zodat min max search kan bepalen wat het beste is
    def evaluate_position(self, board: chess.Board) -> float:
        #schaakmat
        if board.is_checkmate():
            return -1e9
        # 3x dezelfde zet spelen
        if board.is_repetition(3):
            return -1e6
        # 2x zelfde zet
        if board.is_repetition(2):
            return -2.0

        score = 0.0
        # telt voor beide spelers alle stukken op het bord en geeft score
        for piece_type, val in PIECE_VALUES.items():
            if piece_type == chess.KING:
                continue
            score += len(board.pieces(piece_type, board.turn)) * val
            score -= len(board.pieces(piece_type, not board.turn)) * val

        # wie heeft stukken in het centrum. ik = hogere score, anders lager
        for sq in [chess.E4, chess.D4, chess.E5, chess.D5]:
            piece = board.piece_at(sq)
            if piece:
                score += 0.3 if piece.color == board.turn else -0.3

        # hoeveel zetten heb ik
        score += board.legal_moves.count() * 0.05

        # in begin en midden van het spel
        if len(board.piece_map()) > self.cfg.endgame_threshold:
            our_king   = board.king(board.turn)
            their_king = board.king(not board.turn)

            if our_king is not None:
                score -= bin(int(board.attacks_mask(our_king))).count("1") * 0.05
            if their_king is not None:
                score += bin(int(board.attacks_mask(their_king))).count("1") * 0.05

        # als stukken niet meer op begin plek staan, score omhoog! en voor tegenstander andersom
        knight_start = {chess.WHITE: {chess.G1, chess.B1}, chess.BLACK: {chess.G8, chess.B8}}
        bishop_start = {chess.WHITE: {chess.F1, chess.C1}, chess.BLACK: {chess.F8, chess.C8}}

        for color, sign in [(board.turn, 1.0), (not board.turn, -1.0)]:
            for sq in board.pieces(chess.KNIGHT, color):
                if sq not in knight_start[color]:
                    score += sign * 0.2
            for sq in board.pieces(chess.BISHOP, color):
                if sq not in bishop_start[color]:
                    score += sign * 0.2

        # pionnen over de flanken zijn blijkbaar lastiger te verdedigen dus als die er staan score naar beneden
        flank_squares = {chess.A3, chess.A4, chess.H3, chess.H4}
        for sq in flank_squares:
            piece = board.piece_at(sq)
            if piece and piece.piece_type == chess.PAWN:
                score -= 0.15 if piece.color == board.turn else -0.15
        # nu kunnen we de beste zet kiezen
        return score

    # pak de beste zet en return die
    def best_model_move(self, board: chess.Board) -> chess.Move:
        ranked_moves = self.rank_moves(board, top_k=1)
        best_move = ranked_moves[0][0]
        return best_move

    # verzamel alle legale zetten en laat ze score door mijn transformer
    def rank_moves(self, board: chess.Board, top_k: int) -> List[Tuple[chess.Move, float]]:
        #haal legale zetten op
        legal = list(board.legal_moves)
        if not legal:
            return []
        # prompt maken en zetten omzetten naar uci
        prompt     = self.make_prompt(board.fen())
        move_ucis  = [mv.uci() for mv in legal]
        # hier worden de scores gegeven
        raw_scores = self.score_moves(prompt, move_ucis)

        scored = []
        for mv, score in zip(legal, raw_scores):
            bonus    = 0.0
            # kijken welk stuk slaat en wordt geslagen
            victim   = board.piece_at(mv.to_square)
            attacker = board.piece_at(mv.from_square)

            # bij positieve trade hogere score
            if board.is_capture(mv):
                if victim and attacker:
                    v_val = PIECE_VALUES.get(victim.piece_type, 0)
                    a_val = PIECE_VALUES.get(attacker.piece_type, 0)
                    if v_val > a_val:
                        bonus += 0.5
                    elif v_val == a_val:
                        bonus += 0.1

                elif victim is None:
                    bonus += 0.05

            # hogere bonus als de zet schaak geeft
            board.push(mv)
            if board.is_check():
                bonus += 0.3
            board.pop()

            # pion promoveren is nog beter
            if mv.promotion is not None:
                bonus += 0.7

            scored.append((mv, score + bonus))
        # scores opslaan en sorteren en beste k terug geven
        scored.sort(key=lambda x: x[1], reverse=True)
        best_moves = scored[:max(1, top_k)]
        return best_moves

    @staticmethod
    def make_prompt(fen: str) -> str:
        prompt = f"FEN: {fen}\nBest move (UCI):"
        return prompt

    def score_moves(self, prompt: str, moves: List[str]) -> List[float]:
        self.tokenizer.padding_side = "left"

        # tekst maken
        texts  = [prompt + " " + m for m in moves]

        # tokenize de tekst
        inputs = self.tokenizer(
            texts,
            return_tensors = "pt",
            padding = True,
            truncation = True,
            max_length = self.MAX_LEN,
        ).to(self.device)

        # lengte van prompt berekenen
        p_len = len(self.tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=self.MAX_LEN,
        )["input_ids"])
        # transformer gebruiken om te voorspelen welke token erna komt
        with torch.inference_mode():
            logits = self.model(**inputs).logits

        # maak er log probabilities van
        log_probs = torch.log_softmax(logits, dim=-1)
        token_ids = inputs["input_ids"][:, 1:].unsqueeze(-1)
        gathered  = log_probs[:, :-1].gather(-1, token_ids).squeeze(-1)

        # padding tokens niet mee laten tellen
        mask = inputs["attention_mask"][:, 1:].float()

        # prompt negeren
        mask[:, : min(p_len, gathered.shape[1])] = 0.0

        # bereken uiteindelijke scores
        scores = (gathered * mask).sum(dim=-1)
        return scores.tolist()