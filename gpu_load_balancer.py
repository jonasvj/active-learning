import time
import subprocess


def get_pending_jobs(queues):
    bstat = subprocess.run(['bstat'], capture_output=True)
    bstat_output = bstat.stdout.decode()

    pending_jobs = {queue: [] for queue in queues}
    
    for line in bstat_output.split('\n'):
        
        if line.startswith('JOBID'):
            continue
        
        line = line.split()
        if len(line) != 0:
            jobid, queue, status = int(line[0]), line[2], line[5]

            if status == 'PEND' and queue in queues:
                pending_jobs[queue].append(jobid)
    
    # Sort by jobid
    for job_list in pending_jobs.values():
        job_list.sort(reverse=True)

    return pending_jobs


def switch_to_queue(jobid, queue):
    subprocess.run(['bswitch', queue, str(jobid)])
    print(f'Moved job {jobid} to queue {queue}.')


def main():
    print('Starting GPU load balancer.')
    
    queues = ['gpuv100', 'gpua100', 'gpua10', 'gpua40']
    pending_jobs = get_pending_jobs(queues)
    
    empty_queues = [q for q in queues if len(pending_jobs[q]) == 0]
    full_queues = [q for q in queues if len(pending_jobs[q]) > 1]

    while len(full_queues) != 0:
        
        # Move a job from a full queue to an empty queue
        if len(empty_queues) != 0:
            empty_queue = empty_queues[0]
            full_queue = full_queues[-1]
            
            job = pending_jobs[full_queue].pop()
            switch_to_queue(job, empty_queue)
        
        time.sleep(30)
        pending_jobs = get_pending_jobs(queues)
        empty_queues = [q for q in queues if len(pending_jobs[q]) == 0]
        full_queues = [q for q in queues if len(pending_jobs[q]) > 1]
    
    print('Finished! All queues now have 1 or less pending jobs.')


if __name__ == '__main__':
    main()