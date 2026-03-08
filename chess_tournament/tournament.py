import time
import random
from typing import List, Dict, Any
from .game import Game
from .players import EnginePlayer
import gc
import importlib.util
import traceback
from pathlib import Path
import sys

def instantiate_participant(desc: Dict[str, Any]):
    """
    desc is a lightweight descriptor:
      - student: {"type":"student","id": "...", "name": "...", "repo_path": "/content/student_submissions/12345"}
      - baseline: {"type":"baseline","id":"baseline-key","name":"Name","factory": callable}
    Returns a Player instance or raises an Exception with diagnostics.
    """
    if desc.get("type") == "baseline":
        # baseline: call the factory (should return a Player instance)
        factory = desc.get("factory")
        if not callable(factory):
            raise RuntimeError(f"Baseline descriptor {desc.get('id')} missing callable factory")
        return factory()

    if desc.get("type") == "student":
        repo_path = Path(desc.get("repo_path", ""))
        player_py = repo_path / "player.py"
        if not player_py.exists():
            raise FileNotFoundError(f"player.py not found for student {desc.get('id')} at {player_py}")

        # import under a unique name so multiple students can be loaded in same process sequentially
        module_name = f"student_player_{desc.get('id')}_{int(time.time()*1000)}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, str(player_py))
            if spec is None or spec.loader is None:
                raise ImportError("could not create spec/loader")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        except Exception as e:
            tb = traceback.format_exc()
            raise ImportError(f"Failed to import {player_py} for {desc.get('id')}: {e}\n{tb}")

        # look for TransformerPlayer
        cls = getattr(mod, "TransformerPlayer", None)
        if cls is None:
            raise AttributeError(f"TransformerPlayer class not found in {player_py} for {desc.get('id')}")

        # try common ctor patterns
        try:
            try:
                inst = cls(desc.get("name"))
            except TypeError:
                inst = cls()
        except Exception as e:
            tb = traceback.format_exc()
            raise RuntimeError(f"Failed to instantiate TransformerPlayer for {desc.get('id')}: {e}\n{tb}")

        return inst

    raise ValueError(f"Unknown descriptor type: {desc!r}")


def destroy_instance(inst):
    """
    Try to release memory used by a Player instance:
    - delete common large attributes (model, tokenizer, pipe)
    - del the object, run GC
    - if torch is available and cuda used, call empty_cache()
    """
    try:
        # try to delete attributes typically used by model wrappers
        for attr in ("model", "tokenizer", "pipe", "llm", "tokenizer_"):
            try:
                if hasattr(inst, attr):
                    try:
                        delattr(inst, attr)
                    except Exception:
                        try:
                            delattr(inst, attr)
                        except Exception:
                            pass
            except Exception:
                pass
    except Exception:
        pass

    # final cleanup
    try:
        del inst
    except Exception:
        pass

    gc.collect()

    # try to clear torch GPU memory if available
    try:
        import torch
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        # not critical — just best-effort
        pass

def round_robin_tournament(
    players: List,
    games_per_pair: int = 2,
    verbose: bool = False,
    engine_break: float = 3.0,
    engine_break_jitter: float = 1.0,
    max_half_moves: int = 150
) -> Dict:
    """
    Round-robin: every unordered pair plays `games_per_pair` games.
    Returns a summary dict with scores, games_played, fallbacks, leaderboard.
    """

    names = [p.name for p in players]
    scores = {n: 0.0 for n in names}
    fallbacks = {n: 0 for n in names}
    games_played = {n: 0 for n in names}

    n = len(players)
    pairs = []
    # build unordered pairs (i < j)
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((players[i], players[j]))

    total_matches = len(pairs) * games_per_pair
    print(f"🏁 Round-robin: {len(players)} players, {len(pairs)} pairs, {games_per_pair} games/pair -> {total_matches} matches\n")

    match_idx = 0
    for p1, p2 in pairs:
        for g in range(games_per_pair):
            match_idx += 1
            # alternate colors: even g -> p1 white, odd g -> p2 white
            if g % 2 == 0:
                white, black = p1, p2
            else:
                white, black = p2, p1

            #if verbose:
            print(f"> Match {match_idx}/{total_matches}: {white.name} (white) vs {black.name} (black) ... ", end="", flush=True)

            game = Game(p1, p2, max_half_moves=max_half_moves)
            result, match_scores, match_fallbacks = game.play(verbose=verbose, force_colors=(white, black))

            # update stats
            scores[p1.name] += match_scores[p1.name]
            scores[p2.name] += match_scores[p2.name]
            fallbacks[p1.name] += match_fallbacks[p1.name]
            fallbacks[p2.name] += match_fallbacks[p2.name]
            games_played[p1.name] += 1
            games_played[p2.name] += 1

            #if verbose:
            print(f"Result: {result} | Scores: {match_scores} | Fallbacks: {match_fallbacks}")

            # if engine involved, pause a bit
            if isinstance(p1, EnginePlayer) or isinstance(p2, EnginePlayer):
                pause = engine_break + random.uniform(0.0, engine_break_jitter)
                if verbose:
                    print(f"[pause] Waiting {pause:.2f}s before next match")
                time.sleep(pause)

    # leaderboard sort by points, then fewer fallbacks
    def sort_key(nm):
        return (-scores[nm], fallbacks[nm], random.random())

    leaderboard = sorted(names, key=sort_key)

    print("\n🏆 FINAL ROUND-ROBIN LEADERBOARD 🏆")
    print("Rank | Name | Points | Games | Fallbacks")
    for rank, name in enumerate(leaderboard, start=1):
        print(f"{rank:>2} | {name:<15} | {scores[name]:>5.1f} | {games_played[name]:>5} | {fallbacks[name]:>8}")

    return {
        "scores": scores,
        "games_played": games_played,
        "fallbacks": fallbacks,
        "leaderboard": leaderboard
    }

def swiss_tournament(
    participant_descs: List[Dict[str,Any]],
    instantiate_fn,
    destroy_fn,
    n_rounds: int = 3,
    games_per_pairing: int = 2,
    max_half_moves: int = 150,
    engine_break: float = 0.0,
):
    """
    Swiss tournament using PER-MATCH instantiation.
    For odd participant counts, one player receives a 1-point bye each round.
    Final ranking uses deterministic tie-breaks:
      1) points (descending)
      2) Buchholz (sum of faced-opponents' final scores, descending)
      3) fallback count (ascending)
      4) name (ascending)

    participant_descs : lightweight descriptors (students + baselines)
    instantiate_fn    : function(desc) -> Player instance
    destroy_fn        : function(instance)
    """

    names = [p["name"] for p in participant_descs]

    scores = {n: 0.0 for n in names}
    fallbacks = {n: 0 for n in names}
    byes = {n: 0 for n in names}
    # Track opponents once per pairing-round (not per game).
    # Using a list keeps round-level history and allows rematches (if any) to
    # contribute again to Buchholz, which matches per-round semantics.
    opponents = {n: [] for n in names}
    past_pairs = set()

    print(f"🏁 Swiss tournament ({len(names)} players, {n_rounds} rounds)")

    for rnd in range(1, n_rounds + 1):
        print(f"\n=== Round {rnd} ===")

        # Sort by score
        sorted_names = sorted(names, key=lambda n: (-scores[n], random.random()))

        used = set()
        round_pairings = []

        # If odd number of participants, assign one bye:
        # 1) never give a second bye until everyone has one
        # 2) among eligible players, pick the lowest score
        if len(names) % 2 == 1:
            min_byes = min(byes.values())
            bye_candidates = [n for n in names if byes[n] == min_byes]
            bye_player = min(bye_candidates, key=lambda n: (scores[n], n))

            used.add(bye_player)
            byes[bye_player] += 1
            scores[bye_player] += 1.0
            print(f"Bye: {bye_player} (+1.0 point)")

        for i, p1 in enumerate(sorted_names):
            if p1 in used:
                continue

            opponent_found = None
            for p2 in sorted_names[i+1:]:
                if p2 in used:
                    continue
                if frozenset({p1,p2}) not in past_pairs:
                    opponent_found = p2
                    break

            if opponent_found is None:
                for p2 in sorted_names[i+1:]:
                    if p2 not in used:
                        opponent_found = p2
                        break

            if opponent_found:
                round_pairings.append((p1, opponent_found))
                used.add(p1)
                used.add(opponent_found)
                past_pairs.add(frozenset({p1, opponent_found}))

        print("Pairings:", round_pairings)

        # ---- PLAY MATCHES ----
        for p1_name, p2_name in round_pairings:

            desc1 = next(d for d in participant_descs if d["name"] == p1_name)
            desc2 = next(d for d in participant_descs if d["name"] == p2_name)

            # Record this head-to-head once per pairing instead of once per game.
            # This prevents overweighting multi-game pairings in Buchholz.
            opponents[p1_name].append(p2_name)
            opponents[p2_name].append(p1_name)

            for game_idx in range(games_per_pairing):

                print(f"> {p1_name} vs {p2_name} (game {game_idx+1}) ... ", end="")

                p1 = instantiate_fn(desc1)
                p2 = instantiate_fn(desc2)

                try:
                    game = Game(p1, p2, max_half_moves=max_half_moves)
                    result, match_scores, match_fallbacks = game.play(verbose=False)

                finally:
                    destroy_fn(p1)
                    destroy_fn(p2)

                scores[p1_name] += match_scores[p1_name]
                scores[p2_name] += match_scores[p2_name]
                fallbacks[p1_name] += match_fallbacks[p1_name]
                fallbacks[p2_name] += match_fallbacks[p2_name]

                print(f"{result}")

                if engine_break > 0:
                    time.sleep(engine_break)

    # Buchholz = sum of final scores of opponents faced per round.
    # Opponents are tracked at pairing-level, so games_per_pairing does not
    # inflate Buchholz; rematches across rounds (if forced) are counted again.
    buchholz = {n: sum(scores[opp] for opp in opponents[n]) for n in names}

    # ---- FINAL SORT ----
    leaderboard = sorted(
        names,
        key=lambda n: (-scores[n], -buchholz[n], fallbacks[n], n)
    )

    print("\n🏆 FINAL LEADERBOARD 🏆")
    for rank, name in enumerate(leaderboard, start=1):
        print(
            f"{rank:>2}. {name:<20}  {scores[name]:>5.1f} pts"
            f" | buchholz {buchholz[name]:>5.1f}"
            f" | byes {byes[name]} | fallbacks {fallbacks[name]}"
        )

    return {
        "scores": scores,
        "byes": byes,
        "fallbacks": fallbacks,
        "buchholz": buchholz,
        "opponents": {n: list(opponents[n]) for n in names},
        "leaderboard": leaderboard
    }
    
def run_tournament(player_a, player_b, n_games=4, verbose=False, max_half_moves=200):
    results = {
        player_a.name: {"points": 0.0, "wins": 0, "draws": 0, "fallbacks": 0},
        player_b.name: {"points": 0.0, "wins": 0, "draws": 0, "fallbacks": 0},
    }

    print(f"🏁 Tournament: {player_a.name} vs {player_b.name}")
    print(f"Games: {n_games}\n")

    for game_idx in range(1, n_games + 1):
        print(f"--- Game {game_idx} ---")

        game = Game(player_a, player_b, max_half_moves)
        result, scores, fallbacks = game.play(verbose=verbose)

        # Aggregate stats
        for player_name in results.keys():
            results[player_name]["points"] += scores[player_name]
            results[player_name]["fallbacks"] += fallbacks[player_name]

        if result == "1-0":
            winner = max(scores, key=scores.get)
            results[winner]["wins"] += 1

        elif result == "0-1":
            winner = max(scores, key=scores.get)
            results[winner]["wins"] += 1

        else:
            results[player_a.name]["draws"] += 1
            results[player_b.name]["draws"] += 1

        print("Result:", result)
        print("Scores:", scores)
        print("Fallbacks:", fallbacks, "\n")

    # Final summary
    print("\n🏆 FINAL SUMMARY 🏆")

    for player_name, stats in results.items():
        print(f"\n{player_name}")
        print(f"Points: {stats['points']:.1f}")
        print(f"Wins: {stats['wins']}")
        print(f"Draws: {stats['draws']}")
        print(f"Fallbacks used: {stats['fallbacks']}")
