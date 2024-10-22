#include <iostream>
#include "DRPMD.h"
using namespace std;

int main() {
  //Sample 20 SA-rec Garnet Problems
	Sample_sa(5, 4, 3, 100, 0.1, 0.2, 20);
  //Sample_sa(10, 6, 3, 100, 0.1, 0.2, 20);
  //Sample_sa(15, 8, 3, 100, 0.1, 0.2, 20);
  //Sample_sa(20, 10, 8, 50, 0.1, 0.2, 20);
  
  //Sample 20 S-rec Garnet Problems
	Sample_s(20, 10, 8, 100, 0.1, 0.2, 20);
	system("pause");
	return 0;
}
