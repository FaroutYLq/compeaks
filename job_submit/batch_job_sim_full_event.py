import numpy as np
import time
import os, shlex
import sys
#from subprocess import Popen, PIPE, STDOUT, TimeoutExpired
import utilix
from utilix.batchq import *
print(utilix.__file__)

DEFAULT_SIM_FULL_EVENT_RUNS = np.array(['sim_KrS1_full_event_0', 'sim_KrS1_full_event_1', 'sim_KrS1_full_event_2', 
                                        'sim_KrS1_full_event_3', 'sim_KrS1_full_event_4', 'sim_KrS1_full_event_5',
                                        'sim_KrS1_full_event_6', 'sim_KrS1_full_event_7', 'sim_KrS1_full_event_8',
                                        'sim_KrS1_full_event_9'])

class Submit(object):
    '''
        Take maximum number of nodes to use at once
        Submit each group to a node and excute
    '''
    def name(self):
        return self.__class__.__name__

    def execute(self, *args, **kwargs):
        eval('self.{name}(*args, **kwargs)'.format(name = self.name().lower()))

    def submit(self, loop_over=[], max_num_submit=10, nmax=3):
        _start = 0
        self.max_num_submit = max_num_submit
        self.loop_over = loop_over
        self.p = True

        index = _start
        while (index < len(self.loop_over) and index < nmax):
            if (self.working_job() < self.max_num_submit):
                self._submit_single(loop_index=index,
                                    loop_item=self.loop_over[index])

                time.sleep(1.0)
                index += 1

    # check my jobs
    def working_job(self):
        cmd='squeue --user={user} | wc -l'.format(user = 'yuanlq')
        jobNum=int(os.popen(cmd).read())
        return  jobNum -1

    def _submit_single(self, loop_index, loop_item):
        run_id = loop_item
        jobname = 'sim_full_event_%s'%(run_id)

        # Modify here for the script to run
        jobstring = "python /home/yuanlq/xenon/compeaks/job_submit/sim_full_event_peak_extra.py %s"%(run_id)
        print(jobstring)

        # Modify here for the log name
        utilix.batchq.submit_job(
            jobstring, log='/home/yuanlq/.tmp_job_submission/sim_full_event_peak_extra_%s.log'%(run_id), partition='xenon1t', qos='xenon1t',
            account='pi-lgrandi', jobname=jobname,
            delete_file=True, dry_run=False, mem_per_cpu=20000,
            container='xenonnt-development.simg',
            cpus_per_task=1)

p = Submit()

loop_over = DEFAULT_SIM_FULL_EVENT_RUNS

print('Runs to process: ', len(loop_over))

p.execute(loop_over=loop_over, max_num_submit=50, nmax=10000)