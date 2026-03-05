"""
  The General case of ellipsephic sets given by finite automata.

  For a fixed integer b > 1, the original base, let L be a regular
  language whose alphabet if {0,1,...,b-1}, with the restriction that
  all words do not end in 0 and the empty word is not in L.
  For an integer B >= b, define the
  map E_B : L ---> N, by E_B((a_0,...,a_n)) := sum_{j=0}^n a_j B^j.

  Note that if alpha, beta in L, with E_b(alpha) < E_b(beta),
  then E_B(alpha) < E_B(beta).

  For j >= 1 an integer, we define the series
  S_{L,B}(j) := \sum_{alpha in L} 1/E_B(alpha)^j.

  Note that if j > 1 this series always converges. Let D denote the minimal
  DFA which recognizes L, with states s_1,...,.s_m.
  For each d in {0,1,...,b-1} define the m x m, 0/1 matrix ("stepping matrix")
  A^(d)_{i,j} = 1 if there is a transition from s_i to s_j with input d, otherwise 0.
  Set A = sum_d A^(d). Proposition: S_{L,B}(1) converges if and only if
  the largest eigenvalue of of A is < B.

  We generalize a construction of Baillie and Schmelzer.

  If s_i, s_j are states of D, define W(i,j) = words over the alphabet {0,...,b-1}
  which will transit from s_i to s_j.
  U(i, j, k, m) = sum_{w in W(i,j), len(w) = k} 1/E_B(w)^m

  
  
  
"""
pass
