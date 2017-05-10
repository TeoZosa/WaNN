from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

extensions = [
    Extension("*", [r"C:\Users\damon\PycharmProjects\BreakthroughANN\monte_carlo_tree_search\*.pyx"],
        extra_compile_args=[r"/openmp"]),
    # Everything but primes.pyx is included here.
    Extension("*", [r"C:\Users\damon\PycharmProjects\BreakthroughANN\Breakthrough_Player\*.pyx"],
              extra_compile_args=[r"/openmp"]),
    Extension("*", [r"C:\Users\damon\PycharmProjects\BreakthroughANN\tools\*.pyx"],
              extra_compile_args=[r"/openmp"]),
]
setup(
    # name = "My hello app",
    ext_modules = cythonize(extensions),
)
# setup(
#     ext_modules = cythonize(r"C:\Users\damon\PycharmProjects\BreakthroughANN\Breakthrough_Player\breakthrough_player.pyx" )
# )
# setup(
#     ext_modules = cythonize(r"C:\Users\damon\PycharmProjects\BreakthroughANN\Breakthrough_Player\board_utils.pyx" )
# )
# setup(
#     ext_modules = cythonize(r"C:\Users\damon\PycharmProjects\BreakthroughANN\Breakthrough_Player\policy_net_utils.pyx" )
# )
# setup(
#     ext_modules = cythonize(r"C:\Users\damon\PycharmProjects\BreakthroughANN\monte_carlo_tree_search\expansion_MCTS_functions.py" )
# )
# setup(
#     ext_modules = cythonize(r"C:\Users\damon\PycharmProjects\BreakthroughANN\monte_carlo_tree_search\MCTS.pyx" )
# )
# setup(
#     ext_modules = cythonize(r"C:\Users\damon\PycharmProjects\BreakthroughANN\monte_carlo_tree_search\tree_builder.py" )
# )
# setup(
#     ext_modules = cythonize(r"C:\Users\damon\PycharmProjects\BreakthroughANN\monte_carlo_tree_search\tree_search_utils.pyx" )
# )
# setup(
#     ext_modules = cythonize(r"C:\Users\damon\PycharmProjects\BreakthroughANN\monte_carlo_tree_search\TreeNode.py" )
# )
# setup(
#     ext_modules = cythonize(r"C:\Users\damon\PycharmProjects\BreakthroughANN\tools\utils.pyx" )
# )
