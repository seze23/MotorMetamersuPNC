"""Set a MoBL-ARMS pose without sequential constraint-assembly drift.

Evaluate the model's two-point linear couplers and its native wrist spline,
then set all independent/dependent coordinates without intermediate assembly.
Reject unsupported constraint forms rather than approximate them.
"""
import xml.etree.ElementTree as ET
import numpy as np

class CoupledPose:
    def __init__(self,model,model_path):
        self.coordinates=model.getCoordinateSet()
        self.couplings=[]
        root=ET.parse(model_path).getroot()
        for constraint in root.findall('.//CoordinateCouplerConstraint'):
            if constraint.findtext('isEnforced','true').strip()=='false':continue
            dep=constraint.findtext('dependent_coordinate_name').strip()
            independents=constraint.findtext('independent_coordinate_names').split()
            spline=constraint.find('.//SimmSpline')
            if len(independents)!=1 or spline is None:raise ValueError('Unsupported coordinate coupler')
            x=np.fromstring(spline.findtext('x'),sep=' ');y=np.fromstring(spline.findtext('y'),sep=' ')
            if len(x)!=len(y) or len(x)<2:raise ValueError('Invalid spline')
            if len(x)>2:
                import opensim as osm
                function=osm.SimmSpline()
                for a,b in zip(x,y):function.addPoint(float(a),float(b))
                scale=float(constraint.findtext('scale_factor','1'))
                self.couplings.append((dep,independents[0],None,(function,scale)))
                continue
            slope=(y[1]-y[0])/(x[1]-x[0]);offset=y[0]-slope*x[0]
            scale=float(constraint.findtext('scale_factor','1'))
            self.couplings.append((dep,independents[0],float(slope*scale),float(offset*scale)))
    def set_radians(self,state,values):
        values=dict(values)
        for dep,source,slope,offset in self.couplings:
            source_value=values.get(source)
            if source_value is None:source_value=self.coordinates.get(source).getValue(state)
            if slope is None:
                import opensim as osm
                function,scale=offset
                values[dep]=function.calcValue(osm.Vector(1,float(source_value)))*scale
            else:values[dep]=source_value*slope+offset
        for name,value in values.items():self.coordinates.get(name).setValue(state,float(value),False)
