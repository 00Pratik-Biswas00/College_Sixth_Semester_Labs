#include <stdio.h>

// Function to perform Hamming code encoding (4-bit message to 7-bit codeword)
void hamming_encode(int message[4], int codeword[7])
{
  // Calculate parity bits (p1, p2, p3) using even parity
  codeword[0] = (message[0] + message[1] + message[3]) % 2;
  codeword[1] = (message[0] + message[2] + message[3]) % 2;
  codeword[2] = message[0];
  codeword[3] = (message[1] + message[2] + message[3]) % 2;
  codeword[4] = message[1];
  codeword[5] = message[2];
  codeword[6] = message[3];
}

// Function to perform Hamming code decoding (7-bit received codeword to 4-bit message)
void hamming_decode(int received_codeword[7], int decoded_message[4])
{
  // Calculate syndrome bits (s1, s2, s3)
  int s1 = (received_codeword[0] + received_codeword[2] + received_codeword[4] + received_codeword[6]) % 2;
  int s2 = (received_codeword[1] + received_codeword[2] + received_codeword[5] + received_codeword[6]) % 2;
  int s3 = (received_codeword[3] + received_codeword[4] + received_codeword[5] + received_codeword[6]) % 2;

  // Determine error position (0 for no error, 1-7 for error bit position)
  int error_position = (s1 * 1) + (s2 * 2) + (s3 * 4);

  // Correct the error (if any)
  if (error_position > 0)
  {
    printf("Error detected at bit %d. Correcting...\n", error_position);
    received_codeword[error_position - 1] ^= 1; // Flip the erroneous bit
  }
  else
  {
    printf("No error detected.\n");
  }

  // Retrieve the original message bits
  decoded_message[0] = received_codeword[2];
  decoded_message[1] = received_codeword[4];
  decoded_message[2] = received_codeword[5];
  decoded_message[3] = received_codeword[6];
}

int main()
{
  int message[4];         // Example 4-bit message
  int codeword[7];        // Array to store the 7-bit codeword
  int decoded_message[4]; // Array to store the decoded 4-bit message

  // Prompt the user to enter the 4-bit message
  printf("Enter a 4-bit message (e.g., 1011): ");
  if (scanf("%1d%1d%1d%1d", &message[0], &message[1], &message[2], &message[3]) != 4)
  {
    printf("Invalid input. Please enter exactly 4 bits (0 or 1).\n");
    return 1; // Exit with error
  }

  // Encode the message using Hamming code
  hamming_encode(message, codeword);

  // Display the encoded codeword
  printf("Encoded codeword: ");
  for (int i = 0; i < 7; i++)
  {
    printf("%d", codeword[i]);
  }
  printf("\n");

  // Simulate transmission by introducing errors
  codeword[3] ^= 1; // Flip a bit to simulate an error

  // Decode the received codeword using Hamming code
  hamming_decode(codeword, decoded_message);

  // Display the decoded message
  printf("Decoded message: ");
  for (int i = 0; i < 4; i++)
  {
    printf("%d", decoded_message[i]);
  }
  printf("\n");

  return 0;
}

// Output ->

// Enter a 4-bit message (e.g., 1011): 1101
// Encoded codeword: 1010101
// Error detected at bit 4. Correcting...
// Decoded message: 1101