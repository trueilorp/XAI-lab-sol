Explainable AI Laboratory
====================================

This repository contains the materials, exercises, and supplementary resources for the Explainable Artificial Intelligence (XAI) laboratory component of the course. 
The code is intended to be run on an Ubuntu 22.04 machine (a virtual machine is also fine, e.g., with VirtualBox).

------------------------------------
Virtual Machine (VM) setup
------------------------------------

If Ubuntu 22.04 is not native on your machine:
1. Follow the instructions here for Ubuntu VM setup on Virtual Box https://ubuntu.com/tutorials/how-to-run-ubuntu-desktop-on-a-virtual-machine-using-virtualbox#1-overview
2. Download Ubuntu 22 OS image from here https://releases.ubuntu.com/jammy/

------------------------------------
System setup
------------------------------------

To ensure full functionality of the laboratory exercises, please follow the steps below after cloning the repository.

1. Clone the repository
```bash   
    git clone https://github.com/Isla-lab/XAI.git
    cd XAI
```
----------------------------

2. Install required binaries
From the repository root, execute the following script to download and unpack the required binaries for Clingo and ILASP:
``` bash
    bash install_binaries.sh
```
This script will automatically retrieve the appropriate versions of both tools and prepare them for use in subsequent labs.

If necessary, once the installation is completed, ensure that both files are executable:
``` bash
    chmod +x clingo ILASP
```

----------------------------------

3. Move executables to the system path
Upon completion of the installation, move the executables to /usr/local/bin so that they can be accessed system-wide:
``` bash
    sudo mv clingo /usr/local/bin/
    sudo mv ILASP /usr/local/bin/
```

4. Verify the installation

You may confirm that the installation was successful by running:
``` bash
    clingo --version
    ILASP --version
```
If both commands return version information, the setup has been completed successfully.


------------------------------------
Lab-specific requirements
------------------------------------
Additional requirements for each lab lesson will be added to the dedicated directory in a README file.

------------------------------------
Troubleshooting and Support
------------------------------------
If any issues arise during the installation or execution of the laboratory exercises, please contact us: 
- daniele.meli@univr.it
- celeste.veronese@univr.it
