#accuracy
p = clf.predict([xtest])
count = 0
for i in range(0,21000):
   count += 1 
   if p[i]:
   	  print(actual_label[i])
   else:
      print("0")
   
   print("ACCURACY", (count/21000)*100)