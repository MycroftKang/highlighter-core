import setuptools

from highlighter import __version__

setuptools.setup(
    name="mulgyeol-highlighter",
    version=__version__,
    author="Mycroft Kang",
    author_email="taet777@naver.com",
    description="Finds Highlights From Recorded Live Videos.",
    packages=["highlighter", "highlighter.utils"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    install_requires=[
        "tensorflow>=2.4.0",
        "sklearn>=0.0",
        "pandas>=1.2.0",
        "pygit2>=1.4.0",
        "PyYAML>=5.3.1",
        "aiohttp>=3.7.3",
    ],
    python_requires=">=3.7",
)
