#!/usr/bin/env python3

import ROOT


class VertexTask(ROOT.FairTask):
    def __init__(self):
        self.genfit_tracks = None
        self.vertices = None
        self.vertex_factory = None
        super().__init__()


    def Init(self):
        ROOT.gInterpreter.Declare('#include "TGeoMaterialInterface.h"')
        ROOT.gInterpreter.Declare('#include "MaterialEffects.h"')
        ROOT.gInterpreter.Declare('#include "FieldManager.h"')
        ROOT.gInterpreter.Declare('#include "ConstField.h"')
        ROOT.gInterpreter.Declare('#include "GFRaveVertexFactory.h"')
        ioman = ROOT.FairRootManager.Instance()
        ROOT.SetOwnership(ioman, False)
        in_tree = ioman.GetInTree()
        # task_list = ioman.GetInFile().FileHeader.GetListOfTasks()
        # task_list.Print()
        # task_list[0] = ROOT.TObjString("TrackingTask")
        # for task in task_list:
        #     print(task)
        # TODO check whether tracking is in list of tasks of input file or current FairRun?
        try:
            in_tree.ls()
            self.genfit_tracks = in_tree.genfit_tracks
        except AttributeError as e:
            print(f"Caught exception {e}")
            # TODO raise in case this also doesn't work
            out_tree = ioman.GetOutTree()
            self.genfit_tracks = out_tree.genfit_tracks
        self.vertices = ROOT.std.vector("genfit::GFRaveVertex*")()
        if not (out_tree := ioman.GetOutTree()):
            out_tree = in_tree.CloneTree(0)
            ioman.GetSink().SetOutTree(out_tree)
        out_tree.Branch("RAVE_vertices", self.vertices)
        ioman.AddBranchToList("RAVE_vertices")
        self.vertex_factory = ROOT.genfit.GFRaveVertexFactory()
        geo_mat = ROOT.genfit.TGeoMaterialInterface()
        ROOT.genfit.MaterialEffects.getInstance().init(geo_mat)
        bfield = ROOT.genfit.ConstField(0, 0, 0)
        field_manager = ROOT.genfit.FieldManager.getInstance()
        field_manager.init(bfield)
        ROOT.genfit.MaterialEffects.getInstance().setNoEffects()
        self.vertex_factory.setMethod("avr")
        return ROOT.kSUCCESS

    def Exec(self, opt):
        self.vertices.clear()
        if len(self.genfit_tracks) >= 2:
            self.vertex_factory.findVertices(self.vertices, self.genfit_tracks)

    def FinishTask(self):
        self.vertices.clear()
