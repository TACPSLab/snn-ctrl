from glob import glob

from setuptools import setup

pkg_name = 'mapless_navi'

setup(
    name = pkg_name,
    version = '0.0.0',
    packages = [pkg_name],
    data_files = [
        ('share/ament_index/resource_index/packages', ['resource/' + pkg_name]),
        (f'share/{pkg_name}', ['package.xml']),
        (f'share/{pkg_name}/launch', glob('launch/*launch.[pxy][yma]*')),
        (f'share/{pkg_name}/models', glob('models/**/*', recursive=True)),
        (f'share/{pkg_name}/worlds', glob('worlds/**/*', recursive=True)),
    ],
    install_requires = ['setuptools'],
    zip_safe = True,
    maintainer = 'Unmaintained',
    maintainer_email = 'nobody@nowhere.com',
    description = 'TODO: Package description',
    license = 'TODO: License declaration',
    entry_points = {
        'console_scripts': [
        ],
    },
)
