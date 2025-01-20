job_id=$(sbatch --parsable batch.sh "$@")
while ! scontrol show job "$job_id" | grep -q "JobState=RUNNING"; do
    echo "Waiting for job $job_id to start..."
    sleep 5
done

echo "Job $job_id started"

sattach $job_id.0
