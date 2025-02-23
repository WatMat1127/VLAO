# MIN/wB97XD/Def2SVP

0 1
Pd                 0.06324500   -0.25369800   -0.00063900
P                  2.45445900   -0.47210400    0.01551500
Cl                 3.35547938   -1.43583887    1.08175115
Cl                 3.22833305   -0.87433052   -1.43941281
P                  0.57831000    2.04525800   -0.01424300
Cl                 0.37038507    2.91624737    1.42646061
Cl                -0.09370354    3.15862725   -1.10341189
C                  3.20993400    1.18430500    0.34863100
H                  3.22073000    1.30608700    1.43891700
H                  4.24854300    1.19225100    0.00224000
C                  2.38270100    2.30429800   -0.32132000
H                  2.51598400    2.28994900   -1.41020800
H                  2.68043100    3.29466000    0.03832500
C                 -1.97278300    0.08705600   -0.00326400
C                 -2.66531100    0.22509800   -1.21089900
H                 -2.15471500    0.09982100   -2.16170500
C                 -4.03724700    0.50935700   -1.20425500
H                 -4.56788100    0.61058300   -2.14765600
C                 -4.72212900    0.65375300    0.00537700
C                 -4.03018300    0.50868400    1.21070100
H                 -4.55501100    0.60907400    2.15744000
C                 -2.65801500    0.22432900    1.20863600
H                 -2.14142600    0.09865900    2.15617900
C                 -0.42180000   -2.29059000   -0.00110400
H                 -5.78648400    0.87172700    0.00856200
F                  0.72170700   -3.07282200   -0.05661500
F                 -1.17372300   -2.70061000   -1.05950700
F                 -1.07581300   -2.70748000    1.11980700
Options
SubAddExPot=/xxx/yyy/VL_main.py [must be replaced with the absolute path for VL_main.py]
GauProc=32
GauMem=1600
MinFreqValue=50.0
EigenCheck