# Accurate evaluation of Kempner-like series

On September 1, 2024, Neil Sloane posed the following problem to the
*math-fun* list.

> Let $V(n)$ mean: write n in base 10 but read it as if it were
> base 11.  E.g. $V(27) = 2*11 + 7 = 29$. $V(n)$ is `A171397`.  $V$ is
> interesting because although the harmonic series $\sum_n 1/n$
> diverges, it is a classic result that $S_1 = \sum_n 1/V(n)$
> converges.  The decimal expansion of $S_1$
> is in `A375805`, but only 3 decimal places are known.
>
> Second, $V(\text{prime}_n)$ is `A031216`.
> What is $S_2 = \sum 1/V(\text{prime}_n)$?
> Its value is in `A375863`, but only 1 decimal place is known.

In 1914, Kempner noted that the although the series $\sum_n 1/n$
diverges to infinity, the same series, when restricting $n$ to not
involve a particular digit in its base 10 expansion, *did*
converge. His reasoning is immediately generalizable to any integral
base. Sloane's first problem may be understood to be summing the
harmonic series while omitting those terms whose denominators omit the
digit "10" in base 11.

Here we discuss both problems: the first problem is satisfactorially
solved by a method due to Baillie. We also discuss generalizations of
his method in which the configuration of allowed patterns of digits
constitute a *regular language*.

The second problem, involving a sum over primes, is much more
delicate. It does not yield to the method of Baillie. Instead we
reduce its estimation to an evaluation of the sums of value of the
*prime zeta function* at values related to a Fourier expansion of the
characteristic function of intervals derived from leading digits.

