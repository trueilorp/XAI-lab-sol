#!/usr/bin/env bash

install_script_path="$( cd "$(dirname "$0")" ; pwd -P )"
target_install_path="${install_script_path}"


function install_binary {
  binary_url=$1
  binary_tar_file=$2
  binary_folder=$3
  binary_name=$4
  strip_components=$5

  if [[ ! -f "${binary_tar_file}" ]]; then
    echo "Downloading ${binary_name}..."
    wget -nv -O ${binary_tar_file} ${binary_url}
  fi

  if [[ ! -d "${binary_folder}" ]]; then
    echo "Extracting ${binary_tar_file}..."
    mkdir -p ${binary_folder}
    tar -xzf ${binary_tar_file} --strip-components=${strip_components} -C ${binary_folder}
  fi

  echo "Moving binary to src/bin..."
  mkdir -p "${target_install_path}"
  mv "${binary_folder}/${binary_name}" "${target_install_path}"

  echo "Removing downloaded and extracted files..."
  rm -rf ${binary_tar_file} ${binary_folder}

  echo "Success installing ${binary_name}!"
}

ilasp_url=""
ilasp_tar_file="ILASP.tar.gz"
ilasp_folder="ILASP_folder"
ilasp_binary="ILASP"

clingo_url=""
clingo_tar_file="clingo.tar.gz"
clingo_folder="clingo_folder"
clingo_binary="clingo"


# TODO update the URLs to the latest versions  
if [[ "$OSTYPE" == "linux-gnu" ]]; then
  ilasp_url="https://github.com/ilaspltd/ILASP-releases/releases/download/v4.4.1/ILASP-v4.4.1-ubuntu.tar.gz"
  clingo_url="https://github.com/potassco/clingo/releases/download/v5.4.0/clingo-5.4.0-linux-x86_64.tar.gz"
elif [[ "$OSTYPE" == "darwin"* ]]; then
  ilasp_url="https://github.com/ilaspltd/ILASP-releases/releases/download/v4.4.1/ILASP-4.4.1-macOS-M1.tar.gz"
  clingo_url="https://github.com/potassco/clingo/releases/download/v5.4.0/clingo-5.4.0-macos-x86_64.tar.gz"
else
  echo "Error: Only Linux and MacOS installations are supported."
  exit
fi

if [[ ! -f "${target_install_path}/${ilasp_binary}" ]]; then
  install_binary ${ilasp_url} ${ilasp_tar_file} ${ilasp_folder} ${ilasp_binary} 0
  sudo mv "${target_install_path}/${ilasp_binary}" /usr/local/bin/
  sudo ln -s /usr/local/bin/${ilasp_binary} /usr/local/bin/ILASP
fi

if [[ ! -f "${target_install_path}/${clingo_binary}" ]]; then
  install_binary ${clingo_url} ${clingo_tar_file} ${clingo_folder} ${clingo_binary} 1
  sudo mv "${target_install_path}/${clingo_binary}" /usr/local/bin/
  sudo ln -s /usr/local/bin/${clingo_binary} /usr/local/bin/clingo
fi
