import numpy as np
import time
import os, shlex
import sys

# from subprocess import Popen, PIPE, STDOUT, TimeoutExpired
import utilix
from utilix.batchq import *

print(utilix.__file__)

_, signal = sys.argv

AR_AVAILABLE = np.array(
    [
        "034160",
        "033781",
        "033492",
        "033492",
        "033582",
        "033823",
        "033841",
        "034145",
        "033555",
        "033573",
        "034211",
        "034076",
        "033995",
        "034163",
        "033540",
        "034157",
        "033802",
        "033781",
        "034301",
        "034013",
        "033959",
        "033995",
        "034235",
        "033790",
        "033488",
        "033564",
        "034274",
        "034142",
        "034280",
        "033475",
        "034250",
        "034214",
        "034262",
        "034148",
        "034301",
        "034121",
        "034292",
        "034097",
        "033519",
        "034028",
        "033841",
        "033501",
        "034070",
        "033591",
        "033745",
        "034250",
        "033579",
        "033796",
        "033826",
        "034016",
    ]
)
KR_AVAILABLE = np.array(
    [
        "018223",
        "018834",
        "030532",
        "030430",
        "030403",
        "023392",
        "030406",
        "018902",
        "018913",
        "025633",
        "033226",
        "023555",
        "018767",
        "029509",
        "018614",
        "031903",
        "018253",
        "018568",
        "028701",
        "027016",
        "018653",
        "018929",
        "028665",
        "018777",
        "025633",
        "021731",
        "018630",
        "030505",
        "019188",
        "018844",
        "018617",
        "018722",
        "018503",
        "018578",
        "019240",
        "021725",
        "030355",
        "028656",
        "018485",
        "023479",
        "018759",
        "033256",
        "030484",
        "024345",
        "021530",
        "023395",
        "030448",
        "027039",
        "026419",
        "018364",
    ]
)


class Submit(object):
    """
    Take maximum number of nodes to use at once
    Submit each group to a node and excute
    """

    def name(self):
        return self.__class__.__name__

    def execute(self, *args, **kwargs):
        eval("self.{name}(*args, **kwargs)".format(name=self.name().lower()))

    def submit(self, loop_over=[], max_num_submit=10, nmax=3):
        _start = 0
        self.max_num_submit = max_num_submit
        self.loop_over = loop_over
        self.p = True

        index = _start
        while index < len(self.loop_over) and index < nmax:
            if self.working_job() < self.max_num_submit:
                self._submit_single(loop_index=index, loop_item=self.loop_over[index])

                time.sleep(1.0)
                index += 1

    # check my jobs
    def working_job(self):
        cmd = "squeue --user={user} | wc -l".format(user="yuanlq")
        jobNum = int(os.popen(cmd).read())
        return jobNum - 1

    def _submit_single(self, loop_index, loop_item):
        jobname = "peak_extra_%s_%s" % (signal, loop_index)
        run_id = loop_item
        # Modify here for the script to run
        jobstring = (
            "python /home/yuanlq/xenon/compeaks/job_submit/process_peak_extra.py %s %s"
            % (run_id, signal)
        )
        print(jobstring)

        # Modify here for the log name
        utilix.batchq.submit_job(
            jobstring,
            log="/home/yuanlq/.tmp_job_submission/peak_extra_%s_%s.log"
            % (run_id, signal),
            partition="xenon1t",
            qos="xenon1t",
            account="pi-lgrandi",
            jobname=jobname,
            delete_file=True,
            dry_run=False,
            mem_per_cpu=10000,
            container="xenonnt-development.simg",
            cpus_per_task=1,
        )


p = Submit()

# Modify here for the runs to process

loop_overs = {"ArS1": AR_AVAILABLE, "KrS1B": KR_AVAILABLE, "KrS1A": KR_AVAILABLE}
loop_over = loop_overs[signal]

print("Runs to process: ", len(loop_over))

p.execute(loop_over=loop_over, max_num_submit=50, nmax=10000)
