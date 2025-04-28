# PyEOS: Python Equation of State Library

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

PyEOS is a Python package for Equation of State (EOS) calculations, particularly useful in high-energy physics, astrophysics, and material science simulations. It provides tools for working with various equation of state models for thermodynamic calculations.

## Features

- Abstract base class for implementing equation of state models
- Analytical equation of state implementations (e.g., ideal gamma-law)
- Modifiers to extend or combine existing equation of state models
- Writers for exporting equation of state data to standard formats (e.g., SESAME)
- Support for both scalar and array inputs using NumPy
- Electron-ion splitting using the Thomas-Fermi ionization model

## Installation

```bash
pip install pyeos
```

## Requirements

- Python 3.10 or higher
- NumPy 1.20.0 or higher
- SciPy 1.7.0 or higher

## Basic Usage

### Creating an Equation of State

```python
from pyeos.analytical import IdealGamma

# Create an ideal gamma-law EOS for hydrogen
# Parameters: gamma (adiabatic index), A (atomic mass), Z (atomic number)
eos = IdealGamma(5/3, 1.008, 1)

# Calculate thermodynamic properties
density = 1.0  # g/cm³
temperature = 1000.0  # K

internal_energy = eos.InternalEnergyFromDensityTemperature(density, temperature)
pressure = eos.PressureFromDensityTemperature(density, temperature)
helmholtz = eos.HelmholtzFreeEnergyFromDensityTemperature(density, temperature)

print(f"Internal Energy: {internal_energy} erg/g")
print(f"Pressure: {pressure} dyne/cm²")
print(f"Helmholtz Free Energy: {helmholtz} erg/g")
```

### Using Modifiers

```python
from pyeos.analytical import IdealGamma
from pyeos.modifiers import ScaledEos, ZSplit

# Create a base EOS
base_eos = IdealGamma(5/3, 1.008, 1)

# Create a scaled EOS that multiplies all outputs by a scaling function
def scale_fn(rho, temperature):
    return 2.0  # Double all outputs

scaled_eos = ScaledEos(base_eos, scale_fn)

# Split an EOS into electron and ion components
electron_eos, ion_eos = ZSplit(base_eos)
```

### Working with Arrays

```python
import numpy as np
from pyeos.analytical import IdealGamma

eos = IdealGamma(5/3, 1.008, 1)

# Create arrays of density and temperature values
density_array = np.geomspace(1e-6, 10, 100)  # g/cm³
temperature_array = np.geomspace(1, 1e6, 100)  # K

# Create a 2D grid of values
density_grid, temperature_grid = np.meshgrid(density_array, temperature_array)

# Calculate properties for the entire grid at once
energy_grid = eos.InternalEnergyFromDensityTemperature(density_grid, temperature_grid)
pressure_grid = eos.PressureFromDensityTemperature(density_grid, temperature_grid)
```

### Writing to SESAME Format

```python
from pyeos.analytical import IdealGamma
from pyeos.modifiers import ZSplit
from pyeos.writer.sesame import SesameWriter

# Create EOS objects
eos = IdealGamma(5/3, 1.008, 1)
ion_eos, electron_eos = ZSplit(eos)

# Write to SESAME format
with SesameWriter("hydrogen.sesame", 9999) as writer:
    writer.add_comment("Hydrogen equation of state data")
    writer.write(eos, ion_eos, electron_eos)
```

## API Reference

### Core Classes

#### `Eos` (Abstract Base Class)

The base class for all equation of state models. Defines methods for calculating thermodynamic properties.

- `InternalEnergyFromDensityTemperature(rho, temperature)`: Calculate internal energy
- `PressureFromDensityTemperature(rho, temperature)`: Calculate pressure
- `HelmholtzFreeEnergyFromDensityTemperature(rho, temperature)`: Calculate Helmholtz free energy
- `A`: Property returning the atomic mass number
- `Z`: Property returning the atomic number

### Analytical Models

#### `IdealGamma`

Implements an ideal gas equation of state with a constant adiabatic index (gamma).

```python
IdealGamma(gamma, A, Z)
```

- `gamma`: Adiabatic index (ratio of specific heats)
- `A`: Atomic mass number
- `Z`: Atomic number

### Modifiers

#### `ScaledEos`

Wraps an existing equation of state and applies a scaling function to its outputs.

```python
ScaledEos(eos, scale_fn)
```

- `eos`: The base equation of state to be scaled
- `scale_fn`: A function that takes density and temperature and returns a scaling factor

#### `ZSplit`

Splits an equation of state into electron and ion components based on the Thomas-Fermi ionization model.

```python
electron_eos, ion_eos = ZSplit(eos)
```

- `eos`: The equation of state to split

### Writers

#### `SesameWriter`

Writes equation of state data to the SESAME format.

```python
SesameWriter(file_name, material_id)
```

- `file_name`: Path to the output file
- `material_id`: Material ID for the SESAME file

Methods:

- `add_comment(comment)`: Add a comment to the SESAME file
- `write(eos, ion_eos, electron_eos)`: Write complete equation of state data

## Types

- `EOSReal`: Type for values that can be either scalar floats or numpy arrays of float64
- `EOSArray`: Type for numpy arrays of float64 used in EOS calculations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
