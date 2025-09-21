# C# Neural Network Implementation

C# implementation of the 2-layer neural network for cross-language verification.

## Files

- `ann.sln` - Visual Studio solution file
- `ann/Program.cs` - Main neural network implementation
- `ann/ann.csproj` - C# project file
- `ann/Program.exe` - Compiled executable

## Network Specification

- **Input**: [0.05, 0.10]
- **Hidden**: 2 neurons, sigmoid activation
- **Output**: 2 neurons, sigmoid activation
- **Target**: [0.01, 0.99]
- **Learning Rate**: 0.5
- **Iterations**: 10,000 (default)

## Usage

### Using .NET CLI
```bash
cd ann
dotnet run
```

### Using Visual Studio
1. Open `ann.sln`
2. Build solution (Ctrl+Shift+B)
3. Run project (F5)

### Manual Compilation
```bash
cd ann
csc Program.cs
Program.exe
```

## Implementation Details

- **Update Method**: Parallel weight updates
- **Language**: C# with imperative programming style
- **Output**: Detailed step-by-step calculations
- **Purpose**: Cross-language algorithm verification

## Key Features

- Step-by-step output showing all calculations
- Detailed weight and bias tracking
- Error progression display
- Traditional imperative programming approach
- Cross-platform compatibility with .NET Core

## Prerequisites

- .NET Framework 4.0+ or .NET Core 2.0+
- Visual Studio (optional, recommended for development)

## Expected Results

The C# implementation should produce results consistent with other implementations, demonstrating that the backpropagation algorithm works identically across programming languages when using the same mathematical approach.