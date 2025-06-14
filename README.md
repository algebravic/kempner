Accurate evaluation of Kempner-like series
==========================================

In 1914 Kempner noted that the although the series
$`\sum_{n=1}^\infty \frac{1}{n}`$
diverges to infinity, that the same
series when restricting $n$ to not involve a particular digit in its
base 10 expansion *did* converge. Recently, Neil Sloane posed the
following problem:

> Let $V(n)$ mean: write n in base 10 but read it as if it were
base 11.  E.g. $V(27) = 2*11 + 7 = 29$. $V(n)$ is `A171397`.  $V$ is
interesting because although the harmonic series $`\sum_n 1/n`$
diverges, it is a classic result that $`S_1 = \sum_n 1/V(n)`$
converges.  The decimal expansion of $`S_1`$
is in `A375805`, but only 3 decimal places are known.

> Second, $`V(\text{prime}_n)`$ is `A031216`.
> What is $`S_2 = \sum 1/V(\text{prime}_n)`$?
> Its value is in `A375863`, but only 1 decimal place is known.
> Could someone calculate $`S_1`$ and $`S_2`$ more accurately?
> As William Cheswick always says, "If brute force doesn't work, use
> more brute force".

Here we discuss both problems: the first problem is satisfactorially
solved by a method due to Baillie. We also discuss generalizations of
his method. The second problem, involving a sum over primes, is much
more delicate. It does not yield to the method of Baillie. Instead we
reduce its estimation to an evaluation of the sums of value of the
*prime zeta function* at values related to a Fourier expansion of the
characteristic function of intervals derived from leading digits.

