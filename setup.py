from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

# extensions = [
#     Extension("*", ["/Users/TeofiloZosa/PycharmProjects/BreakthroughANN/monte_carlo_tree_search/*.pyx"],
#         extra_compile_args=["-Ofast, march=corei7-avx"]),
#     # Everything but primes.pyx is included here.
#     Extension("*", ["/Users/TeofiloZosa/PycharmProjects/BreakthroughANN/Breakthrough_Player/*.pyx"],
#         extra_compile_args=["-Ofast, march=corei7-avx"]),
#     Extension("*", ["/Users/TeofiloZosa/PycharmProjects/BreakthroughANN/tools/*.pyx"],
#         extra_compile_args=["-Ofast, march=corei7-avx"]),
# ]
# setup(
#     # name = "My hello app",
#     ext_modules = cythonize(extensions),
# )
setup(
    ext_modules = cythonize("/Users/TeofiloZosa/PycharmProjects/BreakthroughANN/Breakthrough_Player/breakthrough_player.pyx" )
)
setup(
    ext_modules = cythonize("/Users/TeofiloZosa/PycharmProjects/BreakthroughANN/Breakthrough_Player/board_utils.pyx" )
)
setup(
    ext_modules = cythonize("/Users/TeofiloZosa/PycharmProjects/BreakthroughANN/Breakthrough_Player/policy_net_utils.pyx" )
)
setup(
    ext_modules = cythonize("/Users/TeofiloZosa/PycharmProjects/BreakthroughANN/monte_carlo_tree_search/expansion_MCTS_functions.pyx" )
)
setup(
    ext_modules = cythonize("/Users/TeofiloZosa/PycharmProjects/BreakthroughANN/monte_carlo_tree_search/MCTS.pyx" )
)
setup(
    ext_modules = cythonize("/Users/TeofiloZosa/PycharmProjects/BreakthroughANN/monte_carlo_tree_search/tree_builder.pyx" )
)
setup(
    ext_modules = cythonize("/Users/TeofiloZosa/PycharmProjects/BreakthroughANN/monte_carlo_tree_search/tree_search_utils.pyx" )
)
setup(
    ext_modules = cythonize("/Users/TeofiloZosa/PycharmProjects/BreakthroughANN/monte_carlo_tree_search/TreeNode.pyx" )
)
setup(
    ext_modules = cythonize("/Users/TeofiloZosa/PycharmProjects/BreakthroughANN/tools/utils.pyx" )
)
