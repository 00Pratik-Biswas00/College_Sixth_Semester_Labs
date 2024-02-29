#include <stdio.h>
#include <math.h>

int binaryToDecimal(long long n)
{
  int decimalNumber = 0, i = 0, remainder;

  while (n != 0)
  {
    remainder = n % 10;                     // Extract the last digit
    n /= 10;                                // Remove the last digit
    decimalNumber += remainder * pow(2, i); // Multiply digit by 2 raised to the power of its position
    ++i;                                    // Move to the next position
  }

  return decimalNumber;
}

int calculateOddParity(int num)
{
  int parity = 0;

  while (num)
  {
    parity ^= (num & 1); // XOR operation to toggle the parity
    num >>= 1;           // Shift right to check the next bit
  }

  return parity;
}

int main()
{
  long long binaryNumber;

  printf("Enter a binary number: ");
  scanf("%lld", &binaryNumber);

  int num = binaryToDecimal(binaryNumber);

  int parity = calculateOddParity(num);

  printf("Single-bit odd parity for %d (%lld) is %d\n", num, binaryNumber, parity);

  return 0;
}