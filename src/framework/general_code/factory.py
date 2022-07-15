"""
Abstract Factory Class
that takes a dict as input.
"""

from typing import Dict


class Factory:
    def __init__(self,
                 products: Dict,
                 name="Factory",
                 ):
        self.name = name.title()  # Mostly for debugging reasons.
        self.product_dict = products
        # Usually of type Dict[str, Builder], where Builder is a callable (that returns an object).
        self.subfactories = []

    def __call__(self, product_name):
        for subfactory in self.subfactories:
            if subfactory.has(product_name):
                return subfactory(product_name)

        self.assert_has(product_name)
        return self.product_dict[product_name]

    def __str__(self):
        outstr = f"Factory \"{self.name}\" with the following products:"
        outstr += "".join([f"\n   {ln}" for ln in self.product_dict.keys()])
        if len(self.subfactories) > 0:
            outstr += "\n" + f"as well as {len(self.subfactories)} subfactories."
        return outstr

    def __getitem__(self, item):  # After I made that mistake twice..
        return self(item)

    @property
    def available_products_lines(self):
        apls = [f"{self.name:}:"]
        for key in self.product_dict.keys():
            apls.append(f"  {key}")
        for subfactory in self.subfactories:
            for sf_line in subfactory.available_products_lines:
                apls.append(f"  {sf_line}")
        return apls

    def add_subfactory(self, subfactory, safe_add=False):
        if safe_add:
            for key in subfactory.product_dict.keys():
                assert not self.has(key), \
                    f"Product \"{key}\" already exists, " \
                    f"cannot add the following subfactory to \"{self.name}\" Factory:\n" \
                    f"{subfactory}"
        self.subfactories.append(subfactory)

    def create(self, product_name, *args, **kwargs):
        return self(product_name)(*args, **kwargs)

    def assert_has(self, product_name):
        assert self.has(product_name), (f"I don't know about product with name \"{product_name}\"!" +
                                        f"(Factory: \"{self.name}\")\n" +
                                        f"Available products:\n" +
                                        "\n".join(self.available_products_lines))

    def has(self, product_name):
        for subfactory in self.subfactories:
            if subfactory.has(product_name):
                return True
        return product_name in self.product_dict.keys()
