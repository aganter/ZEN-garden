\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  0 x3 + 0 x5 + x11
Subject To
 c0: x0 - x1 - x2 = 0
 c2: x2 - x4 - x6 = 0
 c5: x4 - 10 x9 = 0
 c6: x6 - 5 x10 = 0
 c7: - x7 + x9 <= -50
 c8: - x8 + x10 <= -100
 c9: x9 + x10 = 1000
Bounds
 x1 = 14500
 x3 = 4500
 x5 = 10000
 x7 = 50
 x8 = 100
 x11 = 0
End
