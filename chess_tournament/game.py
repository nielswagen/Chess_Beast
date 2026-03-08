import chess
import random
import csv
import os
from typing import Optional, Tuple, Dict, List, Any

class Game:
    """
    Orchestrate a match between two Player-like objects.

    play(...):
        - verbose: existing behavior (prints board moves minimally)
        - force_colors: None or (white_player, black_player)
        - log_moves: if True, print a short line per ply and collect a move_log
        - log_to_file: optional path to CSV file to append move records
        - return_move_log: if True, returns (result, scores, fallbacks, move_log)
                           else returns (result, scores, fallbacks)
    """

    def __init__(self, player_a, player_b, max_half_moves: int = 200):
        self.player_a = player_a
        self.player_b = player_b
        self.max_half_moves = max_half_moves

    def _apply_move_with_fallback(self, board: chess.Board, move_str: Optional[str]) -> Tuple[str, bool]:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise RuntimeError("No legal moves available")

        # None or empty -> fallback random legal
        if not move_str:
            fallback = random.choice(legal_moves)
            board.push(fallback)
            return fallback.uci(), True

        # sentinel handled by caller
        if move_str == "__NO_MOVES__":
            return "__NO_MOVES__", False

        # If move_str is a tuple like (move, flag) take first element
        if isinstance(move_str, tuple) and len(move_str) >= 1:
            move_str = move_str[0]

        # parse UCI
        try:
            mv = chess.Move.from_uci(move_str)
        except Exception:
            fallback = random.choice(legal_moves)
            board.push(fallback)
            return fallback.uci(), True

        # legality check
        if mv not in board.legal_moves:
            fallback = random.choice(legal_moves)
            board.push(fallback)
            return fallback.uci(), True

        # legal -> push
        board.push(mv)
        return mv.uci(), False

    def _write_csv_header_if_needed(self, path: str):
        needs_header = not os.path.exists(path) or os.path.getsize(path) == 0
        if needs_header:
            with open(path, "a", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(["ply", "player", "role", "fen_before", "move", "fallback"])

    def _append_move_to_csv(self, path: str, rec: Dict[str, Any]):
        with open(path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow([rec["ply"], rec["player"], rec["role"], rec["fen"], rec["move"], rec["fallback"]])

    def play(
        self,
        verbose: bool = False,
        force_colors: Optional[Tuple] = None,
        log_moves: bool = False,
        log_to_file: Optional[str] = None,
        return_move_log: bool = False
    ) -> Tuple[Any, Dict[str, float], Dict[str, int]]:
        """
        Play a single game.

        Returns:
          - default: (result, scores, fallbacks)
          - if return_move_log=True: (result, scores, fallbacks, move_log)
        """

        board = chess.Board()

        # determine colors
        if force_colors:
            white, black = force_colors
        else:
            players = [self.player_a, self.player_b]
            random.shuffle(players)
            white, black = players

        # initialize
        fallbacks = {white.name: 0, black.name: 0}
        move_log: List[Dict[str, Any]] = []

        # prepare CSV header if requested
        if log_to_file:
            try:
                self._write_csv_header_if_needed(log_to_file)
            except Exception as e:
                # don't fail the game for file issues; just warn
                print(f"[Game] Warning: failed to prepare log file {log_to_file}: {e}")
                log_to_file = None

        if verbose:
            print(f"White: {white.name}  vs  Black: {black.name}")
            print(board, "\n")

        for ply in range(self.max_half_moves):
            if board.is_game_over():
                break

            current = white if board.turn == chess.WHITE else black
            role = "white" if current is white else "black"
            fen = board.fen()

            # ask player for move
            try:
                mv_response = current.get_move(fen)
            except Exception as e:
                if verbose:
                    print(f"[{current.name}] get_move crashed: {e}")
                mv_response = None

            # normalize tuple or bare string
            if isinstance(mv_response, tuple) and len(mv_response) >= 1:
                proposed_move = mv_response[0]
            else:
                proposed_move = mv_response

            # handle engine sentinel: immediate terminal
            if proposed_move == "__NO_MOVES__":
                winner = black if current == white else white
                if verbose:
                    print(f"{current.name} reported __NO_MOVES__ -> immediate loss")
                scores = {self.player_a.name: 0.0, self.player_b.name: 0.0}
                scores[winner.name] = 1.0
                result = "1-0" if winner == white else "0-1"
                # no move to log for this ply (engine claims no moves) — but we can still record it:
                rec = {
                    "ply": ply,
                    "player": current.name,
                    "role": role,
                    "fen": fen,
                    "move": "__NO_MOVES__",
                    "fallback": False
                }
                move_log.append(rec)
                if log_moves:
                    print(f"PLY {ply:03d} | {current.name} | {role} | {fen} | {rec['move']} | fallback:{rec['fallback']}")
                if log_to_file:
                    try:
                        self._append_move_to_csv(log_to_file, rec)
                    except Exception:
                        pass
                if return_move_log:
                    return result, scores, fallbacks, move_log
                else:
                    return result, scores, fallbacks

            # apply move (game-level fallback if needed)
            applied_move, parsing_fallback = self._apply_move_with_fallback(board, proposed_move)

            # increment fallback counter only for parsing/legality fallback (game-level)
            if parsing_fallback:
                fallbacks[current.name] += 1

            # record
            rec = {
                "ply": ply,
                "player": current.name,
                "role": role,
                "fen": fen,
                "move": applied_move,
                "fallback": bool(parsing_fallback)
            }
            move_log.append(rec)

            # print if requested
            if log_moves:
                print(f"PLY {ply:03d} | {current.name} | {role} | {fen} | {applied_move} | fallback:{parsing_fallback}")

            # write to CSV
            if log_to_file:
                try:
                    self._append_move_to_csv(log_to_file, rec)
                except Exception:
                    # ignore file write errors
                    pass

            if verbose:
                # keep existing compact board print if verbose True
                print(f"{current.name}: {applied_move}")

        # final result mapping
        raw_result = board.result()
        if raw_result == "*" or raw_result not in ["1-0", "0-1", "1/2-1/2"]:
            raw_result = "1/2-1/2"

        # map scores to player names (white/black)
        scores = {self.player_a.name: 0.0, self.player_b.name: 0.0}
        if raw_result == "1-0":
            scores[white.name] = 1.0
        elif raw_result == "0-1":
            scores[black.name] = 1.0
        else:
            scores[white.name] = 0.5
            scores[black.name] = 0.5

        if log_moves:
            print("Game finished:", raw_result)
            print("Fallback counts:", fallbacks)

        if return_move_log:
            return raw_result, scores, fallbacks, move_log

        return raw_result, scores, fallbacks
