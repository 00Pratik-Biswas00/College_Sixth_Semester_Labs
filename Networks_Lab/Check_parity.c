#include <stdio.h>

int findParity(int n)
{
  int parity = 0;
  while (n)
  {
    parity = !parity; // Flip the parity bit
    n = n & (n - 1);  // Clear the least significant set bit
  }
  return parity;
}

int main()
{
  int num;
  printf("Enter an integer: ");
  scanf("%d", &num);

  printf("Parity of no %d = %s", num, (findParity(num) ? "odd" : "even"));

  return 0;
}