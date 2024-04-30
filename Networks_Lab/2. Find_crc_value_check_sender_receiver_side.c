#include <stdio.h>
#include <string.h>

#define MAX_DATA_LENGTH 100
#define MAX_POLY_LENGTH 10

char data[MAX_DATA_LENGTH];
char check_value[MAX_POLY_LENGTH]; // Maximum length needed is the generator polynomial length
char gen_poly[MAX_POLY_LENGTH];
int data_length;
int i, j;

void XOR()
{
  for (j = 0; j < strlen(gen_poly); j++)
  {
    // Perform XOR operation
    check_value[j] = ((check_value[j] == gen_poly[j]) ? '0' : '1');
  }
}

void crc()
{
  // initializing check_value
  for (i = 0; i < strlen(gen_poly); i++)
    check_value[i] = data[i];
  do
  {
    // check if the first bit is 1 and calls XOR function
    if (check_value[0] == '1')
      XOR();
    // Move the bits by 1 position for the next computation
    for (j = 0; j < strlen(gen_poly) - 1; j++)
      check_value[j] = check_value[j + 1];
    // appending a bit from data
    check_value[j] = data[i++];
  } while (i <= data_length + strlen(gen_poly) - 1);
  // loop until the data ends
}

int checkErrors()
{
  for (i = 0; i < strlen(gen_poly) - 1; i++)
  {
    // Check if any remainder exists
    if (check_value[data_length + i] == '1')
      return 1; // Error detected
  }
  return 0; // No error detected
}

void receiver()
{
  printf("\nEnter the received data: ");
  scanf("%s", data);
  printf("\nData received: %s\n", data);

  crc();

  int error_detected = checkErrors();
  printf("%s\n\n", (error_detected ? "Error detected" : "No error detected"));
}

int main()
{
  printf("Enter data to be transmitted: ");
  scanf("%s", data);

  printf("\nEnter the generating polynomial: ");
  scanf("%s", gen_poly);

  data_length = strlen(data);

  // Append (n-1) zeros to the data
  for (i = data_length; i < data_length + strlen(gen_poly) - 1; i++)
    data[i] = '0';

  data[data_length + strlen(gen_poly) - 1] = '\0'; // Null-terminate the data

  printf("\nData padded with (n-1) zeros: %s", data);

  crc();

  printf("\nCRC or Check value: %s", check_value);

  // Append data with check_value(CRC)
  strcat(data, check_value);

  printf("\nFinal data to be sent: %s", data);

  receiver();

  return 0;
}

// Output

// Enter data to be transmitted: 1001101

// Enter the generating polynomial: 1011

// Data padded with (n-1) zeros: 1001101000
// CRC or Check value: 101
// Final data to be sent: 1001101000101
// Enter the received data: 1001101101

// Data received: 1001101101
// No error detected
