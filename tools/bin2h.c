#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Convert filename to valid C identifier
void sanitize_name(char* output, const char* input) {
    int i = 0;
    while (*input) {
        if (isalnum(*input)) {
            output[i++] = *input;
        } else {
            output[i++] = '_';
        }
        input++;
    }
    output[i] = '\0';
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input.bin> <output.h>\n", argv[0]);
        fprintf(stderr, "Example: %s weights.bin weights.h\n", argv[0]);
        return 1;
    }

    const char* input_file = argv[1];
    const char* output_file = argv[2];

    // Open input binary file
    FILE* in = fopen(input_file, "rb");
    if (!in) {
        fprintf(stderr, "Error: Cannot open input file '%s'\n", input_file);
        return 1;
    }

    // Get file size
    fseek(in, 0, SEEK_END);
    long file_size = ftell(in);
    fseek(in, 0, SEEK_SET);

    // Calculate number of floats
    size_t num_floats = file_size / sizeof(float);
    if (file_size % sizeof(float) != 0) {
        fprintf(stderr,
                "Warning: File size (%ld bytes) is not a multiple of sizeof(float) (%zu bytes)\n",
                file_size, sizeof(float));
        fprintf(stderr, "Reading %zu complete floats\n", num_floats);
    }

    // Read all floats
    float* weights = (float*)malloc(num_floats * sizeof(float));
    if (!weights) {
        fprintf(stderr, "Error: Failed to allocate memory for %zu floats\n", num_floats);
        fclose(in);
        return 1;
    }

    size_t read_count = fread(weights, sizeof(float), num_floats, in);
    if (read_count != num_floats) {
        fprintf(stderr, "Error: Expected to read %zu floats, got %zu\n", num_floats, read_count);
        free(weights);
        fclose(in);
        return 1;
    }
    fclose(in);

    // Open output header file
    FILE* out = fopen(output_file, "w");
    if (!out) {
        fprintf(stderr, "Error: Cannot create output file '%s'\n", output_file);
        free(weights);
        return 1;
    }

    // Create sanitized array name from input filename
    char array_name[256];
    const char* base_name = strrchr(input_file, '/');
    base_name = base_name ? base_name + 1 : input_file;
    sanitize_name(array_name, base_name);

    // Create header guard
    char guard_name[256];
    snprintf(guard_name, sizeof(guard_name), "%s_H", array_name);
    for (int i = 0; guard_name[i]; i++) {
        guard_name[i] = toupper(guard_name[i]);
    }

    // Write header file
    fprintf(out, "#ifndef %s\n", guard_name);
    fprintf(out, "#define %s\n\n", guard_name);
    fprintf(out, "// Generated from %s\n", input_file);
    fprintf(out, "// Total weights: %zu\n", num_floats);
    fprintf(out, "// File size: %ld bytes\n\n", file_size);

    fprintf(out, "const float %s[%zu] = {\n", array_name, num_floats);

    // Write values (8 per line for readability)
    for (size_t i = 0; i < num_floats; i++) {
        if (i % 8 == 0) {
            fprintf(out, "    ");
        }
        fprintf(out, "%.9ef", weights[i]);
        if (i < num_floats - 1) {
            fprintf(out, ", ");
        }
        if ((i + 1) % 8 == 0 || i == num_floats - 1) {
            fprintf(out, "\n");
        }
    }

    fprintf(out, "};\n\n");
    fprintf(out, "const size_t %s_size = %zu;\n\n", array_name, num_floats);
    fprintf(out, "#endif // %s\n", guard_name);

    fclose(out);
    free(weights);

    printf("Successfully converted %s to %s\n", input_file, output_file);
    printf("Generated array: %s[%zu]\n", array_name, num_floats);

    return 0;
}