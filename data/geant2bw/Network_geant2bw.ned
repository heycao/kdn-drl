package netsimulator;
network  Network_geant2bw
{
    parameters:
        int numNodes = 24;
        int numTx = 24;
    types:
        channel Channel10kbps extends ned.DatarateChannel
        {
            delay = 0ms;
            datarate = 10kbps;
        }
        channel Channel40kbps extends ned.DatarateChannel
        {
            delay = 0ms;
            datarate = 40kbps;
        }
    submodules:
        statistic: Statistic {
            numTx = numTx;
            numNodes = numNodes; 
        } 
        node0: Server {
            id = 0;
            numTx = numTx;
            numNodes = numNodes;
            gates:
                port[2];
        }
        node1: Server {
            id = 1;
            numTx = numTx;
            numNodes = numNodes;
            gates:
                port[4];
        }
        node2: Server {
            id = 2;
            numTx = numTx;
            numNodes = numNodes;
            gates:
                port[3];
        }
        node3: Server {
            id = 3;
            numTx = numTx;
            numNodes = numNodes;
            gates:
                port[4];
        }
        node4: Server {
            id = 4;
            numTx = numTx;
            numNodes = numNodes;
            gates:
                port[2];
        }
        node5: Server {
            id = 5;
            numTx = numTx;
            numNodes = numNodes;
            gates:
                port[2];
        }
        node6: Server {
            id = 6;
            numTx = numTx;
            numNodes = numNodes;
            gates:
                port[4];
        }
        node7: Server {
            id = 7;
            numTx = numTx;
            numNodes = numNodes;
            gates:
                port[3];
        }
        node8: Server {
            id = 8;
            numTx = numTx;
            numNodes = numNodes;
            gates:
                port[8];
        }
        node9: Server {
            id = 9;
            numTx = numTx;
            numNodes = numNodes;
            gates:
                port[5];
        }
        node10: Server {
            id = 10;
            numTx = numTx;
            numNodes = numNodes;
            gates:
                port[2];
        }
        node11: Server {
            id = 11;
            numTx = numTx;
            numNodes = numNodes;
            gates:
                port[4];
        }
        node12: Server {
            id = 12;
            numTx = numTx;
            numNodes = numNodes;
            gates:
                port[5];
        }
        node13: Server {
            id = 13;
            numTx = numTx;
            numNodes = numNodes;
            gates:
                port[3];
        }
        node14: Server {
            id = 14;
            numTx = numTx;
            numNodes = numNodes;
            gates:
                port[2];
        }
        node15: Server {
            id = 15;
            numTx = numTx;
            numNodes = numNodes;
            gates:
                port[2];
        }
        node16: Server {
            id = 16;
            numTx = numTx;
            numNodes = numNodes;
            gates:
                port[2];
        }
        node17: Server {
            id = 17;
            numTx = numTx;
            numNodes = numNodes;
            gates:
                port[3];
        }
        node18: Server {
            id = 18;
            numTx = numTx;
            numNodes = numNodes;
            gates:
                port[3];
        }
        node19: Server {
            id = 19;
            numTx = numTx;
            numNodes = numNodes;
            gates:
                port[2];
        }
        node20: Server {
            id = 20;
            numTx = numTx;
            numNodes = numNodes;
            gates:
                port[2];
        }
        node21: Server {
            id = 21;
            numTx = numTx;
            numNodes = numNodes;
            gates:
                port[3];
        }
        node22: Server {
            id = 22;
            numTx = numTx;
            numNodes = numNodes;
            gates:
                port[2];
        }
        node23: Server {
            id = 23;
            numTx = numTx;
            numNodes = numNodes;
            gates:
                port[2];
        }
        tController: NetTrafficController {
            numTx = numTx;
            numNodes = numNodes; 
            gates:
                out[numTx];
         }

    connections:

      node0.port[0] <--> Channel10kbps <--> node1.port[0];
      node0.port[1] <--> Channel10kbps <--> node2.port[0];
      node1.port[1] <--> Channel40kbps <--> node3.port[0];
      node1.port[2] <--> Channel40kbps <--> node6.port[0];
      node1.port[3] <--> Channel40kbps <--> node9.port[0];
      node2.port[1] <--> Channel10kbps <--> node3.port[1];
      node2.port[2] <--> Channel10kbps <--> node4.port[0];
      node3.port[2] <--> Channel40kbps <--> node5.port[0];
      node3.port[3] <--> Channel10kbps <--> node6.port[1];
      node4.port[1] <--> Channel40kbps <--> node7.port[0];
      node5.port[1] <--> Channel40kbps <--> node8.port[0];
      node6.port[2] <--> Channel40kbps <--> node8.port[1];
      node6.port[3] <--> Channel40kbps <--> node9.port[1];
      node7.port[1] <--> Channel40kbps <--> node8.port[2];
      node7.port[2] <--> Channel40kbps <--> node11.port[0];
      node8.port[4] <--> Channel40kbps <--> node12.port[0];
      node8.port[3] <--> Channel40kbps <--> node11.port[1];
      node8.port[5] <--> Channel40kbps <--> node17.port[0];
      node8.port[6] <--> Channel40kbps <--> node18.port[0];
      node8.port[7] <--> Channel10kbps <--> node20.port[0];
      node9.port[2] <--> Channel10kbps <--> node10.port[0];
      node9.port[3] <--> Channel40kbps <--> node12.port[1];
      node9.port[4] <--> Channel10kbps <--> node13.port[0];
      node10.port[1] <--> Channel10kbps <--> node13.port[1];
      node11.port[2] <--> Channel40kbps <--> node14.port[0];
      node11.port[3] <--> Channel10kbps <--> node20.port[1];
      node12.port[2] <--> Channel10kbps <--> node13.port[2];
      node12.port[3] <--> Channel40kbps <--> node19.port[0];
      node12.port[4] <--> Channel40kbps <--> node21.port[0];
      node14.port[1] <--> Channel10kbps <--> node15.port[0];
      node15.port[1] <--> Channel10kbps <--> node16.port[0];
      node16.port[1] <--> Channel40kbps <--> node17.port[1];
      node17.port[2] <--> Channel40kbps <--> node18.port[1];
      node18.port[2] <--> Channel40kbps <--> node21.port[1];
      node19.port[1] <--> Channel10kbps <--> node23.port[0];
      node21.port[2] <--> Channel10kbps <--> node22.port[0];
      node22.port[1] <--> Channel10kbps <--> node23.port[1];


      tController.out[0] --> { @display("ls=grey,1,d"); } --> node0.tControl;
      tController.out[1] --> { @display("ls=grey,1,d"); } --> node1.tControl;
      tController.out[2] --> { @display("ls=grey,1,d"); } --> node2.tControl;
      tController.out[3] --> { @display("ls=grey,1,d"); } --> node3.tControl;
      tController.out[4] --> { @display("ls=grey,1,d"); } --> node4.tControl;
      tController.out[5] --> { @display("ls=grey,1,d"); } --> node5.tControl;
      tController.out[6] --> { @display("ls=grey,1,d"); } --> node6.tControl;
      tController.out[7] --> { @display("ls=grey,1,d"); } --> node7.tControl;
      tController.out[8] --> { @display("ls=grey,1,d"); } --> node8.tControl;
      tController.out[9] --> { @display("ls=grey,1,d"); } --> node9.tControl;
      tController.out[10] --> { @display("ls=grey,1,d"); } --> node10.tControl;
      tController.out[11] --> { @display("ls=grey,1,d"); } --> node11.tControl;
      tController.out[12] --> { @display("ls=grey,1,d"); } --> node12.tControl;
      tController.out[13] --> { @display("ls=grey,1,d"); } --> node13.tControl;
      tController.out[14] --> { @display("ls=grey,1,d"); } --> node14.tControl;
      tController.out[15] --> { @display("ls=grey,1,d"); } --> node15.tControl;
      tController.out[16] --> { @display("ls=grey,1,d"); } --> node16.tControl;
      tController.out[17] --> { @display("ls=grey,1,d"); } --> node17.tControl;
      tController.out[18] --> { @display("ls=grey,1,d"); } --> node18.tControl;
      tController.out[19] --> { @display("ls=grey,1,d"); } --> node19.tControl;
      tController.out[20] --> { @display("ls=grey,1,d"); } --> node20.tControl;
      tController.out[21] --> { @display("ls=grey,1,d"); } --> node21.tControl;
      tController.out[22] --> { @display("ls=grey,1,d"); } --> node22.tControl;
      tController.out[23] --> { @display("ls=grey,1,d"); } --> node23.tControl;
}