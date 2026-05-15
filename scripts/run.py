from argparse import ArgumentParser
from typing import List
from agents import AllSeeingBanditAgent, PartiallyBlindBanditAgent
from agents.competitors.competitors import (
    CompetitorAgent,
    LinearCompetitorAgent,
    NaiveThresholdCompetitorAgent,
)

class Args:
    def __init__(self) -> None:
        self.budget = 1000
        self.n_timestep = 150

        self.competitor_list = [
            "random",
            "linear",
            "naive",
            "c1"
        ]

        self.lyapunov_V = 2.0
        self.seed = 42
        self.alpha: List[int] = [
            10,
            5,
            10,
            15,
        ]
        self.depletion: List[int] = [
            1,
            1,
            1,
            1,
        ]

        
    @staticmethod
    def add_args(parser: ArgumentParser) -> None:
        parser.add_argument("--budget", type=int, default=1000, help="Total budget")
        parser.add_argument("--n_timestep", type=int, default=150, help="Number of timesteps")
        parser.add_argument("--competitors", type=str, default="random,linear,naive,c1", help="Competitors separated by commas")
        parser.add_argument("--lyapunov_V", type=float, default=2.0, help="Lyapunov parameter")
        parser.add_argument("--seed", type=int, default=42, help="Random seed")
        parser.add_argument("--alpha", type=int, nargs='+', default=[10, 5, 10, 15], help="Hard minimum inventory")
        parser.add_argument("--depletion", type=int, nargs='+', default=[1, 1, 1, 1], help="Depletion per timestep")



def main(args: Args) -> None:


    # Game Loop
    for t in range(args.n_timestep):
        

if __name__ == "__main__":
    parser = ArgumentParser()
    Args.add_args(parser)
    args = parser.parse_args()
    main(args)