# Generate docs with sphinx
cd docs
rm -rf preconditioners
sphinx-apidoc -o preconditioners ../preconditioners
make html
cd ..