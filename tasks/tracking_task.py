#!/usr/bin/env python3

import ROOT
from pat_rec import Track
from track_fit import track_fit, isGood


class TrackingTask(ROOT.FairTask):
    def __init__(self):
        self.track_candidates = None
        self.tracks = None
        self.kalman_fitter = None
        self.Digi_advTargetClusters = None # TODO add to BranchList so that they are found
        self.dependencies = ['PR',]
        super().__init__("TrackingTask")

    def Init(self):
        ROOT.gInterpreter.Declare('#include "TGeoMaterialInterface.h"')
        ROOT.gInterpreter.Declare('#include "MaterialEffects.h"')
        ROOT.gInterpreter.Declare('#include "FieldManager.h"')
        ROOT.gInterpreter.Declare('#include "ConstField.h"')
        ioman = ROOT.FairRootManager.Instance()
        ROOT.SetOwnership(ioman, False)
        in_tree = ioman.GetInTree()
        # self.track_candidates = in_tree.track_candidates
        # self.Digi_advTargetClusters = in_tree.Digi_advTargetClusters
        self.tracks = ROOT.std.vector("genfit::Track*")()
        # if not (out_tree := ioman.GetOutTree()):
        out_tree = in_tree.CloneTree(0)
        ioman.GetSink().SetOutTree(out_tree)
        ioman.RegisterAny["std::vector<genfit::Track*>"]("genfit_tracks", self.tracks, True)
        # out_tree.Branch("genfit_tracks", self.tracks)
        # ioman.AddBranchToList("genfit_tracks")
        print(ioman.CheckBranch("genfit_tracks"))
        geo_mat = ROOT.genfit.TGeoMaterialInterface()
        ROOT.genfit.MaterialEffects.getInstance().init(geo_mat)
        bfield = ROOT.genfit.ConstField(0, 0, 0)
        field_manager = ROOT.genfit.FieldManager.getInstance()
        field_manager.init(bfield)
        ROOT.genfit.MaterialEffects.getInstance().setNoEffects()
        self.kalman_fitter = ROOT.genfit.DAF()
        self.kalman_fitter.setMaxIterations(50)
        # TODO add option for display?
        return ROOT.kSUCCESS

    def Exec(self, opt):
        self.tracks.clear()
        track_id = 0
        for track_candidate in self.track_candidates:
            hits = []
            for i in track_candidate:
                digi_hit = self.Digi_advTargetClusters[i]
                hit = ROOT.Hit()
                hit.det_id = digi_hit.GetDetectorID()
                stop = ROOT.TVector3()
                start = ROOT.TVector3()
                digi_hit.GetPosition(stop, start)
                pos = (stop + start) / 2
                hit.x = pos[0]
                hit.y = pos[1]
                hit.z = pos[2]
                hit.view = int(digi_hit.isVertical())
                hit.hit_id = i
                hits.append(hit)
            track = Track(hits=hits, track_id=track_id)
            fit_track = track_fit(track, fitter=self.kalman_fitter)
            if fit_track and isGood(fit_track):
                self.tracks.push_back(fit_track)
                track_id += 1

    def FinishTask(self):
        self.tracks.clear()
