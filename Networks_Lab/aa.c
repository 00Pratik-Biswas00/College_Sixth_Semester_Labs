#include <stdio.h>
void main()
{
  int d[10];
  int rd[10], c1, c2, c3, c;

  printf("Enter 4-bit binary data : \n");
  scanf("%d", &d[0]);
  scanf("%d", &d[1]);
  scanf("%d", &d[2]);
  scanf("%d", &d[4]);

  d[6] = d[4] ^ d[2] ^ d[0];
  d[5] = d[4] ^ d[1] ^ d[0];
  d[3] = d[2] ^ d[1] ^ d[0];

  printf("\nTransmitted Hamming Code : \n");
  for (int i = 0; i < 7; i++)
    printf("%d", d[i]);

  printf("\nReceived Data : \n");
  for (int i = 0; i < 7; i++)
    scanf("%d", &rd[i]);
  c3 = rd[6] ^ rd[4] ^ rd[2] ^ rd[0];
  c2 = rd[5] ^ rd[4] ^ rd[1] ^ rd[0];
  c1 = rd[3] ^ rd[2] ^ rd[1] ^ rd[0];
  c = (c1 * 4) + (c2 * 2) + c3;

  if (c == 0)
    printf("No error detected");
  else
  {
    printf("Error on position in: %d", c);

    printf("\nData sent : ");
    for (int i = 0; i < 7; i++)
      printf("%d", d[i]);

    printf("\nData received : ");
    for (int i = 0; i < 7; i++)
      printf("%d", rd[i]);

    rd[7 - c] = 1 - rd[7 - c];
    printf("\nCorrect message is ");
    for (int i = 0; i < 7; i++)
      printf("%d", rd[i]);
  }
}