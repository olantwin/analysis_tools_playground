#!/usr/bin/env python3

import ROOT

class PatternRecognitionTask(ROOT.FairTask):
    def __init__(self):
        self.track_candidates = ROOT.std.vector("std::vector<int>")()
        self.Digi_advTargetClusters = None # TODO add to BranchList so that they are found
        self.dependencies = ['Digitisation',]
        self.provides = ['TrackCandidates']
        super().__init__("PatternRecognitionTask")

    def Init(self):
        ioman = ROOT.FairRootManager.Instance()
        ROOT.SetOwnership(ioman, False)
        in_tree = ioman.GetInTree()
        self.track_candidates = ROOT.std.vector("std::vector<int>")()
        ioman.RegisterAny["std::vector<std::vector<int>>"]("track_candidates", self.track_candidates, True)
        self.Digi_advTargetClusters = ioman.GetObject("Digi_advTargetClusters")
        # if not (out_tree := ioman.GetOutTree()):
        out_tree = in_tree.CloneTree(0)
        ioman.GetSink().SetOutTree(out_tree)
        # ioman.AddBranchToList("genfit_tracks")
        # print(ioman.CheckBranch("genfit_tracks"))
        return ROOT.kSUCCESS

    def Exec(self, opt):
        self.track_candidates.clear()

    def FinishTask(self):
        self.track_candidates.clear()
