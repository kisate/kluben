task main()
{

	motor[motorC] = -100; //vacuum
	wait1Msec(4500);

	motor[motorC] = 100; //dut'
	wait1Msec(1000);

	while(true)
		{
    	 motor[motorC] = 100; //dut'
    	 wait1Msec(400);

	     motor[motorC] = 0;   //vipusk
	     wait1Msec(100);
    }
}
