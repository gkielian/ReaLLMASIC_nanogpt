#!/bin/bash

print_help(){
  echo "Usage: $0 [options]
  Print numbers in different bases to files.

  Options:
  -h, --help      Show this help message and exit
  -m, --modulo    Modulo number (default: 128)
"
}

declare -a bases=("1" "2" "4" "8" "16")
data_dir="data"
modulo=128

while [[ $# -gt 0 ]]; do
  case $1 in
    -h|--help)
      print_help
      exit 0
      ;;
    -m|--modulo)
      modulo="$2"
      shift
      ;;
    *)
      echo "Unknown option $1"
      print_help
      exit 1
      ;;
  esac
  shift
done

if [ ! -d "${data_dir}" ]; then
  mkdir -p "${data_dir}"
fi

for i in "${bases[@]}"; do
  echo "$i"
  python print_bases_mod_x.py --modulo "$modulo" --no_separator --base "$i" --seed 16 > "./${data_dir}/base_${i}.txt"
done
