#!/usr/bin/env python3
"""Run sndsw tasks (prototype)."""

import argparse
import logging
import ROOT
from tracking_task import TrackingTask
from vertex_task import VertexTask
from pr_task import PatternRecognitionTask

def main():
    """Run sndsw tasks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-f",
        "--inputfile",
        help="""Simulation results to use as input."""
        """Supports retrieving file from EOS via the XRootD protocol.""",
        required=True,
    )
    parser.add_argument(
        "-g",
        "--geofile",
        help="""Simulation results to use as input. """
        """Supports retrieving files from EOS via the XRootD protocol.""",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--outputfile",
        help="""File to write the filtered tree to."""
        """Will be recreated if it already exists.""",
    )
    parser.add_argument("tasks", help="Tasks to execute.", nargs='+')
    args = parser.parse_args()
    run = ROOT.FairRunAna()
    file_source = ROOT.FairFileSource(args.inputfile)
    run.SetGeomFile(args.geofile)
    run.SetSource(file_source)
    suffix = ""
    tasks = []
    for task in args.tasks:
        match task:
            case "Digi":
                tasks.append(ROOT.DigiTaskSND())
                suffix += "_dig"
            case "PR":
                tasks.append(PatternRecognitionTask())
                suffix += "_PR"
            case "track":
                # ROOT.gSystem.Load("libgenfit2")
                # ROOT.gInterpreter.Declare('#include "Track.h"')
                # ROOT.gInterpreter.Declare('#include "TrackingTask.h"')
                # tasks.append(ROOT.TrackingTask())
                tasks.append(TrackingTask())
                suffix += "_tracked"
            case "vertex":
                tasks.append(VertexTask())
                suffix += "_vertexed"
    if not args.outputfile:
        args.outputfile = args.inputfile.removesuffix(".root") + suffix + ".root"
    outfile = ROOT.FairRootFileSink(args.outputfile)
    rdb = ROOT.FairRuntimeDb.instance()
    rdb.getContainer("FairBaseParSet").setStatic()
    rdb.getContainer("FairGeoParSet").setStatic()
    run.SetSink(outfile)
    for task in tasks:
        # TODO check dependencies
        run.AddTask(task)
    run.Init()
    run.Run()
    # TODO correect task list

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
