export const presets = [
  {
    id: 'nl_prime',
    title: 'NL — isPrime',
    group: 'Natural language',
    description: 'Describe the problem in natural language; the enhancer normalizes it into a prefix-completion prompt.',
    defaultArgs: '7',
    argsHint: 'One integer, e.g. 7',
    prompt: 'Check whether an integer is prime; return true/false',
  },
  {
    id: 'factorial',
    title: 'Method — factorial',
    group: 'Method signature',
    description: 'HumanEval-style prefix: imports + class Problem + open method signature {.',
    defaultArgs: '5',
    argsHint: 'One non-negative integer, e.g. 5',
    prompt: `import java.util.ArrayList;
import java.util.List;

public class Problem {
    /**
     * Return the factorial of a non-negative integer n.
     * Example: factorial(5) returns 120.
     */
    public static long factorial(int n) {`,
  },
  {
    id: 'remove_duplicates',
    title: 'Method — removeDuplicates',
    group: 'Method signature',
    description: 'Remove elements that appear more than once; keep relative order.',
    defaultArgs: '[1, 2, 3, 2, 4]',
    argsHint: 'List<Integer>, e.g. [1, 2, 3, 2, 4]',
    prompt: `import java.util.ArrayList;
import java.util.List;

public class Problem {
    /**
     * From a list of integers, remove all elements that occur more than once.
     * Keep order of elements left the same as in the original list.
     */
    public static List<Integer> removeDuplicates(List<Integer> numbers) {`,
  },
  {
    id: 'right_triangle',
    title: 'Method — rightAngleTriangle',
    group: 'Method signature',
    description: 'Check whether three side lengths form a right triangle.',
    defaultArgs: '3, 4, 5',
    argsHint: 'Three integers a, b, c — e.g. 3, 4, 5',
    prompt: `import java.util.ArrayList;
import java.util.List;

public class Problem {
    /**
     * Given the lengths of the three sides of a triangle, return true if they form
     * a right-angled triangle, false otherwise.
     * Example: rightAngleTriangle(3, 4, 5) returns true.
     */
    public static boolean rightAngleTriangle(int a, int b, int c) {`,
  },
]
