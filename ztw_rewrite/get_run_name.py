# TODO unify arguments in a single file?
from methods.early_exit import parser
from utils import generate_run_name


def main(flags):
    exp_name, run_name = generate_run_name(flags)
    print(exp_name, end='')


if __name__ == "__main__":
    flags = parser.parse_args()
    main(flags)
