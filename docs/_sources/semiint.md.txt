Semi Integration and Differentiation
====================

The following introduction of the fundamentals is (partly) taken from Oldham's good explanations in [1].


Fundamentals
------------------------------------

```{eval-rst}
.. automodule:: ec_tools.semi_integration
```

Call Algorithms
------------------------------------
```{eval-rst}
.. automodule:: ec_tools.semi_integration.semi_integration
   :members:
```
The generalized ``semi_integration`` function can be imported and executed by:  

```sh
from ec_tools.semi_integration import semi_integration
semi_integration(I, t, alg, transonic_backend)
```

Here, the implemented algorithms can be selected by the alg flag (see above) or 

the algorithms can be imported and executed individually (listed below).
Since the semi-integration can be more or less computationally intensive the possibility to speed up the computation by relying on the nice `transonic` library has been build in.


```{eval-rst}
.. automodule:: ec_tools.semi_integration.gruenwald 
   :members:
```

Import the Gruenwald function directly by:

```sh
from ec_tools.semi_integration import gruenwald
res = gruenwald(I, delta_t, v)

```



```{eval-rst}
.. automodule:: ec_tools.semi_integration.riemann
   :members:
```

Import the Riemann function directly by:
```sh
from ec_tools.semi_integration import riemann
res = riemann(I, delta_t, v)

```




```{eval-rst}
.. automodule:: ec_tools.semi_integration.fast_riemann
   :members:
```
Import the Fast Riemann function directly by:
```sh
from ec_tools.semi_integration import fast_riemann
res = fast_riemann(I, delta_t, v)

```


Performance Test
----------

TBA

References
----------

**[1]** K.B. Oldham, *Electrochemical Science and Technology, John Wiley & Sons Ltd, 2012*

**[2]** T. Pajkossy, L. Nyikos, *Fast algorithm for differintegration*, J. Electroanal. Chem. 179, 1984

**[3]** Gruenwald, A.K. Uber, *"begrenzte" Derivationen und deren Anwendungen der Integration und Differentiation.* In Z. Angew. Math. und Phys. 1867, 12, 441-480

**[4]** Riemann, B. et al. in *Versuch einer allgemeinen Auffassung der Integration und Differentiation*, Gesammelte Werke, published posthumously, Teubner, Leipzig, 1892, pp. 353-366
