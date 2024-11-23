import os
import sys
from pathlib import Path
import pkg_resources

def get_installed_packages():
    installed_packages = []
    for dist in pkg_resources.working_set:
        package_path = Path(dist.location) / f'{dist.project_name}-{dist.version}.dist-info'
        if package_path.exists():
            install_time = os.path.getmtime(package_path)
            installed_packages.append((dist.project_name, dist.version, install_time))
    
    installed_packages.sort(key=lambda x: x[2])
    return installed_packages

def write_installed_packages_to_file(output_file):
    installed_packages = get_installed_packages()
    with open(output_file, 'w') as f:
        for name, version, _ in installed_packages:
            f.write(f'{name}=={version}\n')

if __name__ == '__main__':
    output_file = 'installed_packages_in_order.txt'
    write_installed_packages_to_file(output_file)
    print(f'Installed packages have been written to {output_file}')
