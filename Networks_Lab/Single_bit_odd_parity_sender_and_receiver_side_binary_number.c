#include <stdio.h>
#include <string.h>

int countOnes(char binary[])
{
  int count = 0;
  for (int i = 0; i < strlen(binary); i++)
  {
    if (binary[i] == '1')
    {
      count++;
    }
  }
  return count;
}

void addParityBit(char binary[])
{
  int ones_count = countOnes(binary);
  if (ones_count % 2 == 0)
  {
    strcat(binary, "1");
  }
  else
  {
    strcat(binary, "0");
  }
}

void removeParityBit(char binary[])
{
  binary[strlen(binary) - 1] = '\0';
}

int main()
{
  char data[100];
  char received[100];

  printf("Enter the sender side data: ");
  scanf("%s", data);

  addParityBit(data);

  printf("Data with parity bit: %s\n", data);

  printf("Enter the receiver side data: ");
  scanf("%s", received);

  if (strcmp(data, received) == 0)
  {
    printf("No error detected. As the total number of 1 in the receiver side is odd so it's an odd parity. After removing the parity bit the value will be: ");
    removeParityBit(data);
    printf("%s\n", data);
  }
  else
  {
    printf("Error detected.\n");
  }

  return 0;
}

// Output

// Enter the sender side data: 111
// Data with parity bit: 1110
// Enter the receiver side data: 1110
// No error detected. As the total number of 1 in the receiver side is odd so it's an odd parity. After removing the parity bit the value will be: 111