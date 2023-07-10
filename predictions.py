# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 12:53:44 2023

@author: pc
"""
from pydantic import BaseModel

class plantyieldprediction(BaseModel):
    N:int
    P:int 
    K:int
    temperature:float
    humidity:float
    ph:float
    rainfall:float