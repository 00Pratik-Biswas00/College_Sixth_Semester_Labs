#include <stdio.h>
#include <string.h>

// Function to find the One's complement of the given binary string
void Ones_complement(char *data)
{
    while (*data)
    {
        if (*data == '0')
            *data = '1';
        else
            *data = '0';
        data++;
    }
}

// Function to return the checksum value of the given string when divided in K size blocks
void checkSum(char *data, int block_size, char *result)
{
    int n = strlen(data);

    // Check data size is divisible by block_size
    // Otherwise add '0' front of the data
    if (n % block_size != 0)
    {
        int pad_size = block_size - (n % block_size);
        char temp[1000];
        strcpy(temp, data);
        for (int i = 0; i < pad_size; i++)
        {
            data[i] = '0';
        }
        strcpy(data + pad_size, temp);
    }

    // First block of data stored in result variable
    strncpy(result, data, block_size);
    result[block_size] = '\0';

    // Loop to calculate the block wise addition of data
    for (int i = block_size; i < n; i += block_size)
    {
        // Stores the data of the next block
        char next_block[1000];
        strncpy(next_block, data + i, block_size);
        next_block[block_size] = '\0';

        // Stores the binary addition of two blocks
        char additions[1000];
        int sum = 0, carry = 0;

        // Loop to calculate the binary addition of the current two blocks of k size
        for (int k = block_size - 1; k >= 0; k--)
        {
            sum += (next_block[k] - '0') + (result[k] - '0');
            carry = sum / 2;
            additions[k] = (sum % 2) + '0';
            sum = carry;
        }
        additions[block_size] = '\0';

        // After binary add of two blocks with carry, if carry is 1 then apply binary addition
        char final[1000];

        if (carry == 1)
        {
            for (int l = block_size - 1; l >= 0; l--)
            {
                if (carry == 0)
                {
                    final[l] = additions[l];
                }
                else if (((additions[l] - '0') + carry) % 2 == 0)
                {
                    final[l] = '0';
                    carry = 1;
                }
                else
                {
                    final[l] = '1';
                    carry = 0;
                }
            }
            strncpy(result, final, block_size);
        }
        else
        {
            strncpy(result, additions, block_size);
        }
    }

    // Return One's complement of result value
    // which represents the required checksum value
    Ones_complement(result);
}

// Function to check if the received message is same as the sender's message
int checker(char *sent_message, char *rec_message, int block_size)
{
    char sender_checksum[1000];
    char receiver_checksum[1000];

    // Checksum Value of the sender's message
    checkSum(sent_message, block_size, sender_checksum);

    // Checksum value for the receiver's message
    char combined_message[2000];
    sprintf(combined_message, "%s%s", rec_message, sender_checksum);
    checkSum(combined_message, block_size, receiver_checksum);

    // If receiver's checksum value is 0
    if (strstr(receiver_checksum, "00000000") != NULL)
    {
        return 1; // No Error
    }
    else
    {
        return 0; // Error
    }
}

// Driver Code
int main()
{
    char sent_message[] = "10000101011000111001010011101101";
    char recv_message[] = "10000101011000111001010011101101";
    int block_size = 8;

    if (checker(sent_message, recv_message, block_size))
    {
        printf("No Error\n");
    }
    else
    {
        printf("Error\n");
    }

    return 0;
}
