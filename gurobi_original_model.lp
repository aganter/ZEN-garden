\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  2 x7 + 6 x8
Subject To
 c0: x0 - x1 - x2 = 0
 c1: x1 - x3 - x5 = 0
 c2: x2 - x4 - x6 = 0
 c3: x3 - 90 x7 = 0
 c4: x5 - 100 x8 = 0
 c5: x4 - 10 x9 = 0
 c6: x6 - 5 x10 = 0
 c7: - x7 + x9 <= -50
 c8: - x8 + x10 <= -100
 c9: x9 + x10 = 1000
Bounds
 x7 <= 500
 x8 <= 1000
End
