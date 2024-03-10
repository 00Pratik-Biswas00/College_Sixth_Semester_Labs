#include <stdio.h>

// Function to calculate single-bit odd parity
int calculateOddParity(int num)
{
  int parity = 0;

  // Loop through each bit of the number
  while (num)
  {
    parity ^= (num & 1); // XOR operation to toggle the parity
    num >>= 1;           // Shift right to check the next bit
  }

  return parity;
}

// Sender side function to generate single-bit odd parity
int generateOddParity(int data)
{
  int parity = calculateOddParity(data);
  return (data << 1) | parity; // Appending parity bit to the leftmost position
}

// Receiver side function to check single-bit odd parity
int checkOddParity(int received)
{
  int data = received >> 1; // Removing parity bit
  int parity = calculateOddParity(data);

  // Checking if the received parity matches the calculated parity
  return (parity == (received & 1));
}

int main()
{
  int data;

  printf("Enter data to send: ");
  scanf("%d", &data);

  // Sender side
  int senderData = generateOddParity(data);
  printf("Sender: Data with parity: %d\n", senderData);

  // Receiver side
  if (checkOddParity(senderData))
  {
    printf("Receiver: Parity check passed. Data received correctly.\n");
    printf("Receiver: Data without parity: %d\n", senderData >> 1); // Removing parity bit
  }
  else
  {
    printf("Receiver: Parity check failed. Data corrupted.\n");
  }

  return 0;
}
