# PyEOS Examples

This directory contains example scripts demonstrating various features and capabilities of the PyEOS library. These scripts showcase how to use PyEOS for reading, writing, interpolating, and modifying equation of state (EOS) tables in different formats.

## Scripts Overview

### convert_eos_tables.py

**Purpose**: Converts between IONMIX and SESAME equation of state table formats.

**Description**: This script reads an input file in either IONMIX or SESAME format and converts it to the other format, attempting to preserve the grid structure. It automatically detects the input format based on file extension and provides options for specifying atomic properties and material IDs.

**Usage**:

```bash
python convert_eos_tables.py INPUT_FILE OUTPUT_FILE [OPTIONS]
```

**Options**:

- `--input-format [auto|ionmix|sesame]`: Input file format (default: auto-detect)
- `--atomic-number FLOAT`: Atomic number (Z) for IONMIX files
- `--atomic-mass FLOAT`: Atomic mass (A) in AMU for IONMIX files
- `--material-id INT`: Material ID for SESAME files (default: 1000)
- `--twot/--no-twot`: Use two-temperature data (default: --twot)

### ideal_gas_sesame.py

**Purpose**: Generates a SESAME table for an ideal gas with specified atomic properties.

**Description**: This script uses the PyEOS library to create a SESAME table for a material with specified atomic number (Z) and atomic mass number (A) using the ideal gas equation of state with a specified adiabatic index (gamma).

**Usage**:

```bash
python ideal_gas_sesame.py [OPTIONS]
```

**Options**:

- `-z FLOAT`: Atomic number (Z) [required]
- `-a FLOAT`: Atomic mass number (A) [required]
- `--gamma FLOAT`: Adiabatic index (default: 5/3)
- `--material-id INT`: Material ID for SESAME file (default: 9999)
- `--output TEXT`: Output file name (default: z{Z}_a{A}.sesame)

### ionmix_interpolation.py

**Purpose**: Demonstrates the use of IonmixEos for interpolating IONMIX tables.

**Description**: This script shows how to read an IONMIX table file and create an equation of state that interpolates over the tabulated data using xarray. It generates plots of internal energy and pressure over a range of densities and temperatures.

**Usage**:

```bash
python ionmix_interpolation.py IONMIX_FILE [OPTIONS]
```

**Options**:

- `--atomic-number FLOAT`: Atomic number (Z) [required]
- `--atomic-mass FLOAT`: Atomic mass (A) in AMU [required]
- `--component [total|ion|electron]`: Which component to use (default: total)
- `--twot`: Use two-temperature data
- `--rho-min FLOAT`: Minimum density in g/cm³ (default: 1e-8)
- `--rho-max FLOAT`: Maximum density in g/cm³ (default: 100.0)
- `--temp-min FLOAT`: Minimum temperature in K (default: 1e4)
- `--temp-max FLOAT`: Maximum temperature in K (default: 1e8)
- `--points INT`: Number of points in each dimension (default: 100)
- `--output TEXT`: Output file prefix

### sesame_interpolation.py

**Purpose**: Demonstrates the use of SesameEos for interpolating SESAME tables.

**Description**: This script shows how to read a SESAME table file and create an equation of state that interpolates over the tabulated data using xarray. It generates plots of internal energy, pressure, and Helmholtz free energy over a range of densities and temperatures.

**Usage**:

```bash
python sesame_interpolation.py SESAME_FILE [OPTIONS]
```

**Options**:

- `--component [total|ion|electron]`: Which component to use (default: total)
- `--rho-min FLOAT`: Minimum density in g/cm³ (default: 1e-8)
- `--rho-max FLOAT`: Maximum density in g/cm³ (default: 100.0)
- `--temp-min FLOAT`: Minimum temperature in K (default: 1)
- `--temp-max FLOAT`: Maximum temperature in K (default: 1e8)
- `--points INT`: Number of points in each dimension (default: 100)
- `--output TEXT`: Output file prefix

### sesame_z_split.py

**Purpose**: Demonstrates the use of ZSplit to create ion and electron tables.

**Description**: This script loads a SESAME format file, splits the EOS into ion and electron components using the ZSplit modifier, and writes the result to a new SESAME file. This is useful for creating separate ion and electron EOS tables from a total EOS table.

**Usage**:

```bash
python sesame_z_split.py SESAME_FILE [OPTIONS]
```

**Options**:

- `--output TEXT`: Output file name (default: z_split.sesame)

## General Usage Notes

Most of these scripts use the Click library for command-line interfaces, which provides helpful error messages and auto-generated help text. You can get detailed help for any script by running:

```bash
python script_name.py --help
```

Many of the scripts generate plots using Matplotlib. If no output file is specified, the plots will be displayed interactively. If an output file prefix is provided, the plots will be saved to disk with that prefix.
