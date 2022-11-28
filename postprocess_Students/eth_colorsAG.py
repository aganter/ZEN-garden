"""===========================================================================================================================================================================
Title:        ZEN-GARDEN
Created:      May-2022
Authors:      Jacob Mannhardt (jmannhardt@ethz.ch)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:  ETH colors for plots
==========================================================================================================================================================================="""
import matplotlib.colors as mcolors

class ETHColors:

    def __init__(self):
        # load ETH colors
        self.loadColors()
        # set colors
        self.setColors()
        self.setManualColors()

    def retrieveColors(self,inputComponents):
        _listColors = []
        if type(inputComponents) == "str":
            _listColors.append(self.retrieveSpecificColor(inputComponents))
        else:
            for component in inputComponents:
                _listColors.append(self.retrieveSpecificColor(component))
        return _listColors

    def retrieveColorsDict(self,inputComponents):
        _dictColors = dict()
        if type(inputComponents) == "str":
            _dictColors[inputComponents] = self.retrieveSpecificColor(inputComponents)
        else:
            for component in inputComponents:
                _dictColors[component] = self.retrieveSpecificColor(component)
        return _dictColors

    def retrieveSpecificColor(self,component):
        if component in self.colors:
            _color = self.colors[component]
        elif component.replace("_", " ") in self.colors:
            _color = self.colors[component.replace("_", " ")]
        elif component.replace(" ", "_") in self.colors:
            _color = self.colors[component.replace(" ", "_")]
        elif component in self.manualColors:
            _color = self.manualColors[component]
        else:
            _color = self.getColor("blue")
        return _color

    def setColors(self):
        self.colors = {
            "natural gas": self.getColor("blue", 60),
            "dry biomass": self.getColor("green", 40),
            "wet biomass": self.getColor("green",80),
            "electricity": self.getColor("blue", "dark"),
            "biomethane": self.getColor("green", 80),

            "costTotal": self.getColor("blue"),
            "capexTotal": self.getColor("green"),
            "opexTotal": self.getColor("bronze"),
            "costCarrierTotal": self.getColor("red"),
            "costCarbonEmissionsTotal": self.getColor("purple"),

            # hydrogen production technologies
            "H$_\mathrm{2}$ Production": self.getColor("PFblue", "light"),
            "electrolysis": self.getColor("PFblue", "1"),
            "SMR": self.getColor("PFblue", "light"),
            "SMR90": self.getColor("blue", 60),
            "biomethane SMR": self.getColor("PFyellow", "light"),
            "biomethane SMR90": self.getColor("PFyellow", "dark"),
            "gasification": self.getColor("PFgreen", "light"),
            "gasification57": self.getColor("PFgreen", "dark"),

            "anaerobic_digestion": self.getColor("green",80),
            "carbon_storage": self.getColor("grey", 60),
            # conditioning technologies
            "H$_\mathrm{2}$ Conditioning": self.getColor("blue", 60),
            "Conditioning": self.getColor("purple", 60),
            "carbon_liquefication": self.getColor("grey",60),
            "hydrogen_compressor_high": self.getColor("green", 60),
            "hydrogen_liquefication": self.getColor("green", 20),

            # electricity generation
            "renewables": self.getColor("green", 60),
            "wind_onshore": self.getColor("blue"),
            "wind_offshore": self.getColor("blue", "dark"),
            "pv_rooftop": self.getColor("red",60),
            "pv_ground": self.getColor("red", 40),

            # transport
            "H$_\mathrm{2}$ Transport": self.getColor("PFyellow","light"),
            "Biomass Transport": self.getColor("PFgreen", "light"),
            "Transport": self.getColor("bronze", 60),
            # hydrogen transport
            "hydrogen_truck_gas": self.getColor("blue",80),
            "hydrogen_train": self.getColor("blue",60),
            # carbon transport
            "Carbon SC": self.getColor("PFgrey", "light"),
            "carbon_truck": self.getColor("grey",60),
            "carbon_train": self.getColor("grey",40),
            # electricity transmission
            "electricity_transmission": self.getColor("bronze", 60),
            # other
            "Other": self.getColor("grey",80)
        }

    def setManualColors(self):
        self.manualColors = {}
        self.manualColors["Carbon Emission Budget"] = self.getColor("grey","dark")

    def getColor(self,color,shade = 100):
        assert color in self.baseColors, f"color {color} not in base colors. Select from {list(self.baseColors.keys())}."
        assert shade in self.baseColors[color], f"shade {shade} not in shades of color {color}. Select from {list(self.baseColors[color].keys())}."
        return self.hex2rgb(self.baseColors[color][shade])

    def getCustomColormap(self, color, diverging=False, reverse=False):
        """Returns a LinearSegmentedColormap
        :param colorSeq: a sequence of floats and RGB-tuples. The floats should be increasing
        and in the interval (0,1).
        :return cmap: returns colormap"""
        if diverging:
            colorN = (1,1,1)
            color1 = [self.getColor(color[1], shade) for shade in list(self.baseColors[color[1]].keys())]
            color2 = [self.getColor(color[0], shade) for shade in list(self.baseColors[color[0]].keys())]
            color2.reverse()
            colorSeq = color1 + [colorN, 0.5, colorN] + color2
        else:
            colorSeq = [self.getColor(color, shade) for shade in list(self.baseColors[color].keys())]
        if not reverse:
            colorSeq.reverse()

        colorSeq = [(None,) * 3, 0.0] + list(colorSeq) + [1.0, (None,) * 3]
        cdict = {'red': [], 'green': [], 'blue': []}
        for i, item in enumerate(colorSeq):
            if isinstance(item, float):
                r1, g1, b1 = colorSeq[i - 1]
                r2, g2, b2 = colorSeq[i + 1]
                cdict['red'].append([item, r1, r2])
                cdict['green'].append([item, g1, g2])
                cdict['blue'].append([item, b1, b2])
        # LinearSegmentedColormap.from_list("", [(0, "red"), (.1, "violet"), (.5, "blue"), (1.0, "green")])
        return mcolors.LinearSegmentedColormap('CustomMap', cdict)

    def loadColors(self):
        self.baseColors ={}
        self.baseColors["blue"] = {
            "dark": "#08407E",
            100:    "#215CAF",
            80:     "#4D7DBF",
            60:     "#7A9DCF",
            40:     "#A6BEDF",
            20:     "#D3DEEF",
            10:     "#E9EFF7",
        }
        self.baseColors["petrol"] = {
            "dark": "#00596D",
            100: "#007894",
            80: "#3395AB",
            60: "#66AFC0",
            40: "#99CAD5",
            20: "#CCE4EA",
            10: "#E7F4F7",
        }
        self.baseColors["green"] = {
            "dark": "#365213",
            100: "#627313",
            80: "#818F42",
            60: "#A1AB71",
            40: "#C0C7A1",
            20: "#E0E3D0",
            10: "#EFF1E7",
        }
        self.baseColors["bronze"] = {
            "dark": "#956013",
            100: "#8E6713",
            80: "#A58542",
            60: "#BBA471",
            40: "#D2C2A1",
            20: "#E8E1D0",
            10: "#F4F0E7",
        }
        self.baseColors["red"] = {
            "dark": "#96272D",
            100: "#B7352D",
            80: "#C55D57",
            60: "#D48681",
            40: "#E2AEAB",
            20: "#F1D7D5",
            10: "#F8EBEA",
        }
        self.baseColors["purple"] = {
            "dark": "#8C0A59",
            100: "#A30774",
            80: "#B73B92",
            60: "#CA6CAE",
            40: "#DC9EC9",
            20: "#EFD0E3",
            10: "#F8E8F3",
        }
        self.baseColors["grey"] = {
            "dark": "#575757",
            100: "#6F6F6F",
            80: "#8C8C8C",
            60: "#A9A9A9",
            40: "#C5C5C5",
            20: "#E2E2E2",
            10: "#F1F1F1",
        }
        self.baseColors["PFred"] = {
            "dark": "#D84848",
            "dark": "#E58585",
            "light": "#EDB2B2",
            "-1": "#F3CDCD",
        }
        self.baseColors["PFyellow"] = {
            "1": "#EDDE6F",
            "dark":  "#EFDE7B",
            "light": "#F6EEAF",
            "-1":    "#FCF9E0",
        }
        self.baseColors["PFgreen"] = {
            "1": "#2E8E2E",
            "dark": "#80D680",
            "light": "#BBDFBC",
            "-1": "#F0F8EE",
        }
        self.baseColors["PFblue"] = {
            "1": "#1F738D",
            "dark": "#70C6E0",
            "light": "#B4D9E8",
            "-1": "#DFEFF5",
        }
        self.baseColors["PFgrey"] = {
            "1": "#888888",
            "dark": "#B5B5B5",
            "light": "#E2E2E2",
            "-2": "#DBDBDB",
        }
    @staticmethod
    def hex2rgb(hexString, normalized=True):
        if normalized:
            _fac = 255
        else:
            _fac = 1
        hexString = hexString.lstrip('#')
        rgb = tuple(int(hexString[i:i + 2], 16)/_fac for i in (0, 2, 4))
        return rgb
